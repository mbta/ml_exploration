from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class TimestampEncoder(BaseEstimator, TransformerMixin):
    def fit(self, vehicle_datapoints, y=None):
        return self

    def transform(self, vehicle_datapoints, y=None):
        datetimes = vehicle_datapoints['timestamp'].apply(
            lambda timestr: datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        )
        vehicle_datapoints['day_of_week'] = datetimes.apply(
            lambda datetime: datetime.weekday()
        )
        vehicle_datapoints['time_bin'] = datetimes.apply(
            lambda datetime: self._time_bin(datetime)
        )
        return vehicle_datapoints

    def _time_bin(self, datetime):
        return datetime.hour * 4 + datetime.minute // 15
