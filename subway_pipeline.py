import pandas as pd
import os

from actuals_adder import ActualsAdder
from destinations_adder import DestinationsAdder
from location_id_translator import LocationIdTranslator
from locations_adder import LocationsAdder


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

        locations_adder = LocationsAdder(self.locations_path)
        datapoints_with_locations = locations_adder.fit_transform(
            vehicle_datapoints
        )

        destinations_adder = DestinationsAdder(self.locations_path)
        results_with_destinations = destinations_adder.fit_transform(
            datapoints_with_locations
        )

        actuals_adder = ActualsAdder(self.actuals_path)
        datapoints = actuals_adder.fit_transform(results_with_destinations)
        return datapoints.dropna()

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
        return frame_with_terminal_modes
