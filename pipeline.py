import pandas as pd
import os
import functools
from collections import defaultdict


class Pipeline():
    def __init__(
        self,
        actuals_path=os.path.join("datasets", "pa_datapoints.csv"),
        locations_path=os.path.join("datasets", "locations.csv"),
        patterns_path=os.path.join("datasets", "patterns.csv"),
        terminals_path=os.path.join("datasets", "terminal_datapoints.csv"),
        vehicles_path=os.path.join("datasets", "vehicle_datapoints.csv")
    ):
        self.actuals_path = actuals_path
        self.locations_path = locations_path
        self.patterns_path = patterns_path
        self.terminals_path = terminals_path
        self.vehicles_path = vehicles_path

    # Build a dictionary mapping location IDs to GTFS stop IDs, using parent
    # stops where child stops exist.
    def location_id_to_parent_stop_id(self):
        dropped_frame = pd.read_csv(
            self.locations_path,
            usecols=['loc_id', 'gtfs_stop_id']
        )
        dropped_frame = dropped_frame.dropna()
        is_stop = dropped_frame.apply(lambda x: x != "0")
        stops_only = dropped_frame[is_stop].dropna()
        return functools.reduce(self._parent_stop, stops_only.__array__(), {})

    # Build a dataframe mapping each pattern ID to the corresponding terminal
    # stop location ID
    def load_patterns(self):
        dropped_frame = pd.read_csv(
            self.patterns_path, usecols=['pattern_id', 'terminal_stop']
        ).dropna()
        dropped_frame["terminal_stop"] = dropped_frame["terminal_stop"] \
            .apply(lambda str: str.split("|")[0])
        return dropped_frame

    # Build a dataframe mapping each pattern ID to the corresponding terminal
    # stop GTFS ID
    def gtfs_id_for_terminals(self):
        patterns = self.load_patterns()
        locations = self.location_id_to_parent_stop_id()
        patterns["terminal_gtfs_id"] = patterns["terminal_stop"] \
            .apply(lambda loc_id: locations.get(loc_id))
        patterns = patterns.dropna().drop("terminal_stop", axis=1)
        return patterns

    # Build a dataframe with all logged terminal-mode datapoints
    def load_terminal_datapoints(self):
        return pd.read_csv(
            self.terminals_path,
            usecols=[
                'automatic',
                'generation',
                'terminal_stop_id',
                'timestamp'
            ],
            dtype={"terminal_stop_id": str}
        )

    # Build a dataframe with all logged vehicle datapoints
    def load_vehicle_datapoints(self):
        raw_frame = pd.read_csv(
            self.vehicles_path,
            usecols=[
                'current_location_id',
                'generation',
                'gtfs_trip_id',
                'length_of_time_at_current_location',
                'ocs_trip_id',
                'offset_departure_seconds_from_now',
                'pattern_id',
                'timestamp',
                'vehicle_id'
            ]
        )
        terminals = self.gtfs_id_for_terminals()

        frame_with_terminals = pd.merge(
            raw_frame,
            right=terminals,
            left_on="pattern_id",
            right_on="pattern_id",
            how="inner"
        )
        frame_with_terminal_modes = pd.merge(
            frame_with_terminals,
            right=self.load_terminal_datapoints(),
            left_on=["generation", "terminal_gtfs_id"],
            right_on=["generation", "terminal_stop_id"],
            how="inner"
        ).drop(
            ["terminal_stop_id", "timestamp_y", "pattern_id"],
            axis=1
        ).rename({"timestamp_x": "timestamp"}, axis="columns")

        n_hot_locations = self.n_hot_vehicle_locations_by_generation()

        generations = frame_with_terminal_modes["generation"].to_list()

        n_hot_per_row_list_of_lists = list(
            map(
                lambda generation: n_hot_locations[generation], generations
            )
        )
        n_hot_per_row = pd.DataFrame(
            n_hot_per_row_list_of_lists,
            columns=self.all_locs_sorted()
        )
        return frame_with_terminal_modes.join(n_hot_per_row, how="inner")

    # Return array of all location IDs in lexicographical order
    def all_locs_sorted(self):
        loc_frame = pd.read_csv(self.locations_path)
        return sorted(loc_frame["loc_id"].__array__())

    # Builds a map of generations to the list of n-hot encoded vehicle
    # positions for that generation.
    def n_hot_vehicle_locations_by_generation(self):
        raw_frame = pd.read_csv(
            self.vehicles_path,
            usecols=[
                'current_location_id',
                'generation',
                'gtfs_trip_id',
                'length_of_time_at_current_location',
                'ocs_trip_id',
                'offset_departure_seconds_from_now',
                'pattern_id',
                'timestamp',
                'vehicle_id'
            ]
        )
        grouped_frame = raw_frame.groupby(by="generation")

        occupied_locations_by_generation = {}
        generational_n_hot_locations = {}

        for generation in grouped_frame:
            generation_id = generation[0]
            locations = generation[1]["current_location_id"]

            occupied_locations_by_generation[generation_id] = set()
            for location in locations:
                occupied_locations_by_generation[generation_id].add(location)

            generational_n_hot_locations[generation_id] = defaultdict(
                lambda: 0
            )
            for location in occupied_locations_by_generation[generation_id]:
                generational_n_hot_locations[generation_id].update(
                    {location: 1}
                )

        generational_n_hot_location_lists = {}
        for generation in generational_n_hot_locations:
            location_map = generational_n_hot_locations[generation]
            n_hot_columns = []
            for location in self.all_locs_sorted():
                n_hot_columns.append(location_map[location])
                generational_n_hot_location_lists[generation] = n_hot_columns

        return generational_n_hot_location_lists

    def _parent_stop(self, acc, x):
        location_id = x[0]
        gtfs_id = x[1]
        parent_stops = {
            "Alewife-01": "70061",
            "Alewife-02": "70061",
            "Braintree-01": "70105",
            "Braintree-02": "70105",
            "Forest Hills-01": "70001",
            "Forest Hills-02": "70001",
            "Oak Grove-01": "70036",
            "Oak Grove-02": "70036",
            "Government Center-Brattle": "70202"
        }
        parent_id = parent_stops.get(gtfs_id)
        if parent_id:
            gtfs_id = parent_id

        acc[location_id] = gtfs_id
        return acc

    # Build a dataframe with every combination of given vehicle datapoints
    # and possible destination.
    #
    # Note that we don't currently check that the destination makes sense, for
    # example, there will be rows for the Blue Line with a destination of
    # Kendall. Currently these get eliminated in a later step. It might be
    # more efficient to do so up-front here, though we'd then have to have this
    # code know somehow what stops are on what lines.
    def add_all_possible_destinations(self):
        vehicle_datapoints = self.load_vehicle_datapoints()
        gtfs_ids = set(self.location_id_to_parent_stop_id().values())
        blank_frame = pd.DataFrame()
        new_frames = map(
            lambda gtfs_id: self._vehicle_datapoints_with_destination_gtfs_id(
                vehicle_datapoints,
                gtfs_id
            ),
            gtfs_ids
        )
        return blank_frame.append(list(new_frames))

    def _vehicle_datapoints_with_destination_gtfs_id(
        self,
        vehicle_datapoints,
        gtfs_id
    ):
        new_frame = vehicle_datapoints.copy()
        new_frame["destination_gtfs_id"] = str(gtfs_id)
        return new_frame
