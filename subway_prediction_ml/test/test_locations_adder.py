import os
import pandas as pd
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from locations_adder import LocationsAdder

class TestLocationsAdder(unittest.TestCase):
    def test_n_hot_train_locations_added_to_vehicle_datapoints(self):
        locations_data = pd.DataFrame(
            columns=["loc_id"],
            data=[["42--70"], ["43--71"], ["44--72"], ["45--73"]]
        )

        vehicle_data = pd.DataFrame(
            columns=['trip_id', 'generation', 'current_location_id'],
            data=[
                ['B-111', 12345, "42--70"],
                ['B-112', 12345, "44--72"],
                ['B-111', 12346, "43--71"],
                ['B-112', 12346, "45--73"],
                ['B-111', 12347, "44--72"]
            ]
        )

        adder = LocationsAdder(locations_data)
        result = adder.fit_transform(vehicle_data).__array__().tolist()
        assert sorted(result) == [
            ['B-111', 12345, "42--70", 1.0, 0.0, 1.0, 0.0],
            ['B-111', 12346, "43--71", 0.0, 1.0, 0.0, 1.0],
            ['B-111', 12347, "44--72", 0.0, 0.0, 1.0, 0.0],
            ['B-112', 12345, "44--72", 1.0, 0.0, 1.0, 0.0],
            ['B-112', 12346, "45--73", 0.0, 1.0, 0.0, 1.0],
        ]

    def test_locations_can_be_filtered_by_line(self):
        locations_data = pd.DataFrame(
            columns=["loc_id", "line"],
            data=[
                ["42--70", "R"],
                ["43--71", "O"],
                ["44--72", "R"],
                ["45--73", "G"]
            ]
        )

        vehicle_data = pd.DataFrame(
            columns=['trip_id', 'generation', 'current_location_id'],
            data=[
                ['B-111', 12345, "42--70"],
                ['B-112', 12345, "44--72"],
                ['B-111', 12347, "44--72"]
            ]
        )
        adder = LocationsAdder(locations_data, route="R")
        result = adder.fit_transform(vehicle_data).__array__().tolist()
        print(sorted(result))
        assert sorted(result) == [
            ['B-111', 12345, '42--70', 1.0, 1.0],
            ['B-111', 12347, '44--72', 0.0, 1.0],
            ['B-112', 12345, '44--72', 1.0, 1.0]
        ]
