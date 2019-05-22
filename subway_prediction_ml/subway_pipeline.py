import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        locations_adder = LocationsAdder(self.locations_path)
        destinations_adder = DestinationsAdder(self.locations_path)
        actuals_adder = ActualsAdder(self.actuals_path)
        onehot_columns = [
            'current_location_id',
            'terminal_gtfs_id',
            'automatic',
            'destination_gtfs_id',
            'event_type'
        ]
        column_transformer = ColumnTransformer([
            ('1hot', OneHotEncoder(), onehot_columns)
        ], remainder="passthrough")

        pipeline = Pipeline([
            ('terminal_modes_adder', terminal_modes_adder),
            ('locations_adder', locations_adder),
            ('destinations_adder', destinations_adder),
            ('actuals_adder', actuals_adder),
            ('col_transformer', column_transformer),
            ('std_scaler', StandardScaler())
        ])
        return pipeline.fit_transform(vehicle_datapoints)

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
