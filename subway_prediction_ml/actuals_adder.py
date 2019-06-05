import datetime
import pytz

from sklearn.base import BaseEstimator, TransformerMixin


class ActualsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, actuals_frame):
        self.actuals_frame = actuals_frame

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
            on='gtfs_trip_id'
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
        merged_frame = merged_frame.query("actual_seconds_from_now > 0")

        return merged_frame.dropna()

    # Build a dataframe from prediction analyzer logs, dropping actuals without
    # times, commuter rail trips, and duplicates (which we get because we log
    # both from dev-green and prod).
    def _load_actuals(self):
        raw_frame = self.actuals_frame.filter(
            ['event_type', 'stop_id', 'time', 'trip_id', 'vehicle_id'],
            axis=1
        )
        raw_frame['stop_id'] = raw_frame['stop_id'].astype('str')
        dropped_frame = raw_frame.dropna(subset=["time"])
        is_subway = dropped_frame["trip_id"].apply(lambda x: x[0:3] != "CR-")
        return dropped_frame[is_subway].drop_duplicates()
