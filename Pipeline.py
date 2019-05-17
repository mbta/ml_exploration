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

    # When we download CSV data from Splunk, they like to add a bunch of extra
    # columns that we don't need for this purpose.
    splunk_columns = [
        "_raw",
        "_time",
        "app",
        "eventtype",
        "host",
        "index",
        "linecount",
        "product",
        "punct",
        "sourcetype",
        "splunk_server",
        "splunk_server_group",
        "source",
        "tag",
        "tag::eventtype",
        "vendor"
    ]

    def load_actuals(self):
        raw_frame = pd.read_csv(self.actuals_path, dtype={"stop_id": str})
        dropped_frame = raw_frame \
            .dropna(subset=["time"]).drop(self.splunk_columns, axis=1)
        is_subway = dropped_frame["trip_id"].apply(lambda x: x[0:3] != "CR-")
        return dropped_frame[is_subway].drop_duplicates()

    def location_id_to_stop_id(self):
        dropped_frame = pd.read_csv(self.locations_path) \
            .drop([
                "loc_name",
                "line",
                "gtfs_stop_seq",
                "next_loc_ids",
                "bearing",
                "default_rt",
                "latitude",
                "longitude",
                "min_turn_time",
                "nonrevenue",
                "tsp_intersection",
                "tsp_direction",
                "default_pattern"
            ], axis=1)
        dropped_frame = dropped_frame.dropna()
        is_stop = dropped_frame.apply(lambda x: x != "0")
        stops_only = dropped_frame[is_stop].dropna()
        return functools \
            .reduce(self.reduce_function, stops_only.__array__(), {})

    def reduce_function(self, acc, x):
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

    def load_patterns(self):
        dropped_frame = pd.read_csv(self.patterns_path) \
            .drop([
                "reverse_pattern_id",
                "direction_id",
                "direction_name",
                "route_id",
                "ocs_identifiers",
                "first_stop",
                "locations",
                "yard?"
            ], axis=1) \
            .dropna()
        dropped_frame["terminal_stop"] = dropped_frame["terminal_stop"] \
            .apply(lambda str: str.split("|")[0])
        return dropped_frame

    def gtfs_id_for_terminals(self):
        patterns = self.load_patterns()
        locations = self.location_id_to_stop_id()
        patterns["terminal_gtfs_id"] = patterns["terminal_stop"] \
            .apply(lambda loc_id: locations.get(loc_id))
        patterns = patterns.dropna().drop("terminal_stop", axis=1)
        return patterns

    def add_all_possible_destinations(self):
        vehicle_datapoints = self.load_vehicle_datapoints()
        gtfs_ids = set(self.location_id_to_stop_id().values())
        result = pd.DataFrame()
        for gtfs in gtfs_ids:
            new_frame = vehicle_datapoints.copy()
            new_frame["destination_gtfs_id"] = str(gtfs)
            result = result.append(new_frame)
        return result

    def load_terminal_datapoints(self):
        raw_frame = pd.read_csv(
            self.terminals_path, dtype={"terminal_stop_id": str}
        )
        return raw_frame.drop(self.splunk_columns, axis=1)

    def load_vehicle_datapoints(self):
        raw_frame = pd.read_csv(self.vehicles_path)
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
        )
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
        frame_without_splunk_nonsense = frame_with_terminal_modes.drop(
            self.splunk_columns + [
                "terminal_stop_id",
                "timestamp_y",
                "pattern_id"
            ], axis=1
        ).rename({"timestamp_x": "timestamp"}, axis="columns")
        return frame_without_splunk_nonsense.join(n_hot_per_row, how="inner")

    def all_locs_sorted(self):
        loc_frame = pd.read_csv(self.locations_path)
        return sorted(loc_frame["loc_id"].__array__())

    def n_hot_vehicle_locations_by_generation(self):
        raw_frame = pd.read_csv(self.vehicles_path) \
            .drop(self.splunk_columns, axis=1)
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

    def add_actuals(self, predictions):
        actuals = self.load_actuals()
        actuals = actuals.rename(
            {"trip_id": "gtfs_trip_id", "stop_id": "destination_gtfs_id"},
            axis="columns"
        )
        actuals["gtfs_trip_id"] = actuals["gtfs_trip_id"].astype(str)
        predictions["gtfs_trip_id"] = predictions["gtfs_trip_id"].astype(str)
        return predictions.merge(
            actuals,
            how="inner",
            on=["gtfs_trip_id", "destination_gtfs_id"]
        ).drop(
            [
                "generation",
                "gtfs_trip_id",
                "ocs_trip_id",
                "vehicle_id_x",
                "vehicle_id_y",
                "terminal"
            ],
            axis=1
        )
