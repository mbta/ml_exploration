import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OffsetSecondsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, vehicle_datapoints, y=None):
        vehicle_datapoints['offset_departure_seconds_from_now'] = \
            vehicle_datapoints['offset_departure_seconds_from_now'].apply(
                self._encode_offset
            )
        return vehicle_datapoints

    def _encode_offset(self, offset):
        if np.isnan(offset):
            return "nan"
        elif offset < 0:
            return "negative"
        elif offset < 500:
            return "0-500"
        elif offset < 1000:
            return "500-1000"
        elif offset < 1500:
            return "1000-1500"
        elif offset < 2000:
            return "1500-2000"
        elif offset < 2500:
            return "2000-2500"
        elif offset < 3000:
            return "2500-3000"
        elif offset < 3500:
            return "3000-3500"
        elif offset < 4000:
            return "3500-4000"
        elif offset < 4500:
            return "4000-4500"
        elif offset < 5000:
            return "4500-5000"
        else:
            return "5000+"
