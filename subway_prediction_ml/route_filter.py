from sklearn.base import BaseEstimator, TransformerMixin


class RouteFilter(BaseEstimator, TransformerMixin):
    def __init__(self, route):
        self.route = route

    def fit(self, X, y=None):
        return self

    def transform(self, vehicle_datapoints, y=None):
        blue_only = vehicle_datapoints.loc[
            vehicle_datapoints.vehicle_id.map(lambda id: id[0] == self.route)
        ]
        return blue_only.copy()
