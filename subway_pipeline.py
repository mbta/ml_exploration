import pandas as pd
import os

from actuals_adder import ActualsAdder
from destinations_adder import DestinationsAdder
from locations_adder import LocationsAdder
from terminal_modes_adder import TerminalModesAdder


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

        terminal_modes_adder = TerminalModesAdder(
            self.locations_path,
            self.patterns_path,
            self.terminals_path
        )
        datapoints_with_terminal_modes = terminal_modes_adder.fit_transform(
            vehicle_datapoints
        )

        locations_adder = LocationsAdder(self.locations_path)
        datapoints_with_locations = locations_adder.fit_transform(
            datapoints_with_terminal_modes
        )

        destinations_adder = DestinationsAdder(self.locations_path)
        results_with_destinations = destinations_adder.fit_transform(
            datapoints_with_locations
        )

        actuals_adder = ActualsAdder(self.actuals_path)
        datapoints = actuals_adder.fit_transform(results_with_destinations)
        return datapoints.dropna()

    # Build a dataframe with all logged vehicle datapoints
    def _load_vehicle_datapoints(self):
        return pd.read_csv(
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
