import os
import pandas as pd
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from actuals_adder import ActualsAdder


class TestActualsAdder(unittest.TestCase):
    def test_joins_actuals_to_vehicle_datapoints(self):
        actuals_data = pd.DataFrame(
            columns=['trip_id', 'stop_id', 'time'],
            data=[
                ['B-111', '70000', 1559740000],
                ['B-111', '70001', 1559740030],
                ['B-111', '70002', 1559740060],
                ['B-111', '70004', 1559740120],
                ['B-112', '70100', 1559740000],
                ['B-112', '70102', 1559740060],
                ['B-112', '70104', 1559740120]
            ]
        )

        vehicle_data = pd.DataFrame(
            columns=['gtfs_trip_id', 'timestamp'],
            data=[
                ['B-111', "2019-06-05T10:24:30.000000Z"],
                ['B-111', "2019-06-05T10:24:45.000000Z"],
                ['B-112', "2019-06-05T10:24:30.000000Z"],
                ['B-113', "2019-06-05T10:24:30.000000Z"]
            ]
        )

        adder = ActualsAdder(actuals_data)
        result = adder.fit_transform(vehicle_data).__array__().tolist()
        assert sorted(result) == [
            ['B-111', '70000', 9715.0],
            ['B-111', '70000', 9730.0],
            ['B-111', '70001', 9745.0],
            ['B-111', '70001', 9760.0],
            ['B-111', '70002', 9775.0],
            ['B-111', '70002', 9790.0],
            ['B-111', '70004', 9835.0],
            ['B-111', '70004', 9850.0],
            ['B-112', '70100', 9730.0],
            ['B-112', '70102', 9790.0],
            ['B-112', '70104', 9850.0],
        ]

    def test_only_join_data_on_same_service_date(self):
        three_am = 1559804400 # June 6, 2019, 3:00 AM

        actuals_data = pd.DataFrame(
            columns=['trip_id', 'stop_id', 'time'],
            data=[
                ['B-111', '70000', three_am - 1],
            ]
        )

        vehicle_data = pd.DataFrame(
            columns=['gtfs_trip_id', 'timestamp'],
            data=[
                ['B-111', "2019-06-06T07:00:00.000000Z"],
            ]
        )

        adder = ActualsAdder(actuals_data)
        result = adder.fit_transform(vehicle_data).__array__().tolist()
        assert sorted(result) == []
