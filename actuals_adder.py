import datetime
import os
import pandas as pd
import pytz

from sklearn.base import BaseEstimator, TransformerMixin

class ActualsAdder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        actuals_path=os.path.join("datasets", "pa_datapoints.csv")
    ):
        self.actuals_path = actuals_path

    def fit(self, X, y=None):
        return self

    # Join actuals to a set of vehicle datapoints
    def transform(self, vehicle_datapoints, y=None):
        actuals = self._load_actuals()
        actuals = actuals.rename(
            {"trip_id": "gtfs_trip_id", "stop_id": "destination_gtfs_id"},
            axis="columns"
        )
        actuals["gtfs_trip_id"] = actuals["gtfs_trip_id"].astype(str)
        vehicle_datapoints["gtfs_trip_id"] = \
            vehicle_datapoints["gtfs_trip_id"].astype(str)

        merged_frame = vehicle_datapoints.merge(
            actuals,
            how="inner",
            on=["gtfs_trip_id", "destination_gtfs_id"]
        ).drop(
            [
                "generation",
                "gtfs_trip_id",
                "ocs_trip_id",
                "vehicle_id_x",
                "vehicle_id_y"
            ],
            axis=1
        )

        merged_frame["timestamp"] = merged_frame["timestamp"].apply(
            lambda str: datetime.datetime.strptime(
                str,
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=pytz.UTC).timestamp()
        )
        merged_frame["actual_seconds_from_now"] = \
            merged_frame["time"] - merged_frame["timestamp"]
        merged_frame = merged_frame.drop(["timestamp", "time"], axis=1)

        return merged_frame

    # Build a dataframe from prediction analyzer logs, dropping actuals without
    # times, commuter rail trips, and duplicates (which we get because we log
    # both from dev-green and prod).
    def _load_actuals(self):
        raw_frame = pd.read_csv(
            self.actuals_path,
            usecols=['event_type', 'stop_id', 'time', 'trip_id', 'vehicle_id'],
            dtype={"stop_id": str}
        )
        dropped_frame = raw_frame.dropna(subset=["time"])
        is_subway = dropped_frame["trip_id"].apply(lambda x: x[0:3] != "CR-")
        return dropped_frame[is_subway].drop_duplicates()
