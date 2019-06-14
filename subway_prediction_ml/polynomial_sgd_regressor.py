from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor


class PolynomialSGDRegressor(BaseEstimator):
    def __init__(self):
        self.pf = PolynomialFeatures()
        self.sgd = SGDRegressor()

    def fit(self, X, y=None):
        self.sgd.fit(self.pf.fit_transform(X, y), y)
        return self

    def partial_fit(self, X, y=None):
        self.sgd.partial_fit(self.pf.fit_transform(X, y), y)

    def predict(self, X):
        return self.sgd.predict(self.pf.transform(X))
