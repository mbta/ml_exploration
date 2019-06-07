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

        vehicle_datapoints["gtfs_trip_id"] = \
            vehicle_datapoints["gtfs_trip_id"].astype(str)
        vehicle_datapoints["timestamp"] = \
            vehicle_datapoints["timestamp"].apply(
                lambda str: datetime.datetime.strptime(
                    str,
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ).replace(tzinfo=pytz.UTC).timestamp()
            )
        self._add_service_date(actuals, 'time')
        self._add_service_date(vehicle_datapoints, 'timestamp')

        merged_frame = vehicle_datapoints.merge(
            actuals,
            how="inner",
            on=['gtfs_trip_id', 'service_date']
        )
        merged_frame["actual_seconds_from_now"] = \
            merged_frame["time"] - merged_frame["timestamp"]
        merged_frame = merged_frame.drop(
            [
                "timestamp",
                "time",
                'service_date'
            ], axis=1
        )
        merged_frame = merged_frame.query("actual_seconds_from_now > 0")

        return merged_frame.dropna()

    def _add_service_date(self, dataframe, series_name):
        dataframe['service_date'] = dataframe[series_name].apply(
            self._service_date_for_timestamp
        )
        return dataframe

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
        subway_actuals = dropped_frame[is_subway].drop_duplicates()
        subway_actuals = subway_actuals.rename(
            {"trip_id": "gtfs_trip_id", "stop_id": "destination_gtfs_id"},
            axis="columns"
        )
        subway_actuals["gtfs_trip_id"] = \
            subway_actuals["gtfs_trip_id"].astype(str)
        return subway_actuals

    def _service_date_for_timestamp(self, timestamp):
        # Service days run 3 AM to 3 AM, so adjust the time by 3 hours
        adjusted_timestamp = timestamp - 10800
        utc_adjusted_time = datetime. \
            datetime. \
            utcfromtimestamp(adjusted_timestamp). \
            replace(tzinfo=pytz.UTC)
        adjusted_time = utc_adjusted_time.astimezone(
            pytz.timezone('America/New_York')
        )
        return adjusted_time.toordinal()
