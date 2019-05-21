import pandas as pd
import os
from collections import defaultdict

from actuals_adder import ActualsAdder
from destinations_adder import DestinationsAdder
from location_id_translator import LocationIdTranslator


class SubwayPipeline():
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

    def load(self):
        vehicle_datapoints = self._load_vehicle_datapoints()
        destinations_adder = DestinationsAdder(self.locations_path)
        results_with_destinations = destinations_adder.fit_transform(
            vehicle_datapoints
        )
        actuals_adder = ActualsAdder(self.actuals_path)
        datapoints = actuals_adder.fit_transform(results_with_destinations)
        return datapoints.dropna()

    # Return array of all location IDs in lexicographical order
    def _all_locs_sorted(self):
        loc_frame = pd.read_csv(self.locations_path)
        return sorted(loc_frame["loc_id"].__array__())

    # Build a dataframe mapping each pattern ID to the corresponding terminal
    # stop GTFS ID
    def _gtfs_id_for_terminals(self):
        patterns = self._load_patterns()
        locations = LocationIdTranslator(self.locations_path)
        patterns["terminal_gtfs_id"] = patterns["terminal_stop"] \
            .apply(lambda loc_id: locations.get(loc_id))
        patterns = patterns.dropna().drop("terminal_stop", axis=1)
        return patterns

    # Build a dataframe mapping each pattern ID to the corresponding terminal
    # stop location ID
    def _load_patterns(self):
        dropped_frame = pd.read_csv(
            self.patterns_path, usecols=['pattern_id', 'terminal_stop']
        ).dropna()
        dropped_frame["terminal_stop"] = dropped_frame["terminal_stop"] \
            .apply(lambda str: str.split("|")[0])
        return dropped_frame

    # Build a dataframe with all logged terminal-mode datapoints
    def _load_terminal_datapoints(self):
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
    def _load_vehicle_datapoints(self):
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
        terminals = self._gtfs_id_for_terminals()

        frame_with_terminals = pd.merge(
            raw_frame,
            right=terminals,
            left_on="pattern_id",
            right_on="pattern_id",
            how="inner"
        )
        frame_with_terminal_modes = pd.merge(
            frame_with_terminals,
            right=self._load_terminal_datapoints(),
            left_on=["generation", "terminal_gtfs_id"],
            right_on=["generation", "terminal_stop_id"],
            how="inner"
        ).drop(
            ["terminal_stop_id", "timestamp_y", "pattern_id"],
            axis=1
        ).rename({"timestamp_x": "timestamp"}, axis="columns")

        n_hot_locations = self._n_hot_vehicle_locations_by_generation()

        generations = frame_with_terminal_modes["generation"].to_list()

        n_hot_per_row_list_of_lists = list(
            map(
                lambda generation: n_hot_locations[generation], generations
            )
        )
        n_hot_per_row = pd.DataFrame(
            n_hot_per_row_list_of_lists,
            columns=self._all_locs_sorted()
        )
        return frame_with_terminal_modes.join(n_hot_per_row, how="inner")

    # Builds a map of generations to the list of n-hot encoded vehicle
    # positions for that generation.
    def _n_hot_vehicle_locations_by_generation(self):
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
            for location in self._all_locs_sorted():
                n_hot_columns.append(location_map[location])
                generational_n_hot_location_lists[generation] = n_hot_columns

        return generational_n_hot_location_lists
