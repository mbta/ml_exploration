import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from subway_pipeline import SubwayPipeline


class TestSubwayPipeline(object):
    def test_load(self):
        p = SubwayPipeline(
            actuals_path=self._fixture_path('pa_datapoints.csv'),
            locations_path=self._fixture_path('locations.csv'),
            patterns_path=self._fixture_path('patterns.csv'),
            terminals_path=self._fixture_path('terminal_datapoints.csv'),
            vehicles_path=self._fixture_path('vehicle_datapoints.csv')
        )
        print(p.load())
        assert 2 + 2 == 5

    def _fixture_path(self, filename):
        return os.path.abspath(
            os.path.join(__file__, '..', 'datasets', filename)
        )
