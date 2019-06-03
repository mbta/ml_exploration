import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from actuals_adder import ActualsAdder
from locations_adder import LocationsAdder
from offset_seconds_encoder import OffsetSecondsEncoder
from route_filter import RouteFilter
from terminal_modes_adder import TerminalModesAdder
from timestamp_encoder import TimestampEncoder


class SubwayPipeline():
    def __init__(
        self,
        actuals_path=os.path.join("datasets", "pa_datapoints.csv"),
        locations_path=os.path.join("datasets", "locations.csv"),
        patterns_path=os.path.join("datasets", "patterns.csv"),
        terminals_path=os.path.join("datasets", "terminal_datapoints.csv"),
        vehicles_path=os.path.join("datasets", "vehicle_datapoints.csv")
    ):
        self.vehicles_path = vehicles_path

        self.actuals_frame = pd.read_csv(actuals_path)
        self.locations_frame = pd.read_csv(locations_path)
        self.patterns_frame = pd.read_csv(patterns_path)
        self.terminals_frame = pd.read_csv(terminals_path)

    def load(self):
        vehicle_datapoints = self._load_vehicle_datapoints()

        route_filter = RouteFilter('B')
        timestamp_encoder = TimestampEncoder()
        offset_seconds_encoder = OffsetSecondsEncoder()
        terminal_modes_adder = TerminalModesAdder(
            self.locations_frame,
            self.patterns_frame,
            self.terminals_frame
        )
        locations_adder = LocationsAdder(self.locations_frame)
        actuals_adder = ActualsAdder(self.actuals_frame)
        auto_onehot_columns = [
            'current_location_id',
            'terminal_gtfs_id',
            'automatic',
            'destination_gtfs_id',
            'event_type',
        ]

        auto_onehot_encoder = OneHotEncoder()
        offset_onehot_encoder = OneHotEncoder(
            categories=[
                [
                    "nan",
                    "negative",
                    "0-500",
                    "500-1000",
                    "1000-1500",
                    "1500-2000",
                    "2000-2500",
                    "2500-3000",
                    "3000-3500",
                    "3500-4000",
                    "4000-4500",
                    "4500-5000",
                    "5000+"
                ]
            ],
        )
        final_transformer = ColumnTransformer([
            ('pass', 'passthrough', ['actual_seconds_from_now']),
            ('auto1hot', auto_onehot_encoder, auto_onehot_columns),
            (
                'offset1hot',
                offset_onehot_encoder,
                ['offset_departure_seconds_from_now']
            )
        ], remainder=StandardScaler())

        pipeline = Pipeline([
            ('route_filter', route_filter),
            ('timestamp_encoder', timestamp_encoder),
            ('offset_seconds_encoder', offset_seconds_encoder),
            ('terminal_modes_adder', terminal_modes_adder),
            ('locations_adder', locations_adder),
            ('actuals_adder', actuals_adder),
            ('final_transformer', final_transformer),
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
