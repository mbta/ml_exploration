import os
import sklearn
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from subway_pipeline import SubwayPipeline


class TestSubwayPipeline(unittest.TestCase):
    def setUp(self):
        # In theory the unittest.mock library allows you to mock out an
        # instance method on all instances of a given class, but I haven't been
        # able to get it to work. In this test, we want a mock StandardScaler
        # because the effects of that scaler are difficult to calculate and
        # it's a standard (and hopefully well-tested) component already. Hence
        # the monkeypatch, which we'll have to undo later.

        self.original_scaler_method = \
            sklearn.preprocessing.StandardScaler.fit_transform
        sklearn.preprocessing.StandardScaler.fit_transform = (
            lambda self, X, y=None: X
        )

    def tearDown(self):
        sklearn.preprocessing.StandardScaler.fit_transform = \
            self.original_scaler_method

    def test_load(self):
        p = SubwayPipeline(
            actuals_path=self._fixture_path('pa_datapoints.csv'),
            locations_path=self._fixture_path('locations.csv'),
            patterns_path=self._fixture_path('patterns.csv'),
            terminals_path=self._fixture_path('terminal_datapoints.csv'),
            vehicles_path=self._fixture_path('vehicle_datapoints.csv')
        )
        sorted_result = sorted(p.load().tolist())
        print(sorted_result)
        assert sorted_result == [
            [15.617650985717773, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [15.617650985717773, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]

    def _fixture_path(self, filename):
        return os.path.abspath(
            os.path.join(__file__, '..', 'datasets', filename)
        )
