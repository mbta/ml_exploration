import os
import pandas as pd
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from terminal_modes_adder import TerminalModesAdder


class TestTerminalModesAdder(unittest.TestCase):
    def test_transform_adds_pertinent_terminal_modes(self):
        locations_data = pd.DataFrame(
            columns=['loc_id', 'gtfs_stop_id'],
            data=[
                ['42.413524--70.991670', '70060'],  # Wonderland EB
                ['42.407762--70.992530', '70058'],  # Revere Beach EB
                ['42.397213--70.992580', '70056'],  # Beachmont EB
                ['42.358892--71.057401', '70041'],  # State WB
                ['42.359477--71.059658', '70039'],  # Government Center WB
                ['42.361154--71.062084', '70838'],  # Bowdoin WB
            ]
        )

        patterns_data = pd.DataFrame(
            columns=['pattern_id', 'terminal_stop'],
            data=[
                [8, '42.413524--70.991670'],  # Wonderland EB
                [10, '42.361154--71.062084'],  # Bowdoin WB
            ]
        )

        terminals_data = pd.DataFrame(
            columns=[
                'terminal_stop_id',
                'generation',
                'timestamp',
                'automatic'
            ],
            data=[
                [70059, 12345, 'terminal_timestamp1', True],
                [70038, 12345, 'terminal_timestamp2', False],
                [70059, 12344, 'terminal_timestamp3', False],
                [70038, 12344, 'terminal_timestamp4', True]
            ]
        )

        vehicle_data =  pd.DataFrame(
            columns=['trip_id', 'pattern_id', 'generation', 'timestamp'],
            data=[
                ['B-111', 8, 12345, 'vehicle_timestamp1'],
                ['B-112', 8, 12345, 'vehicle_timestamp2'],
                ['B-111', 8, 12344, 'vehicle_timestamp3'],
                ['B-112', 8, 12344, 'vehicle_timestamp4'],
                ['B-211', 10, 12345, 'vehicle_timestamp5'],
                ['B-212', 10, 12345, 'vehicle_timestamp6'],
                ['B-211', 10, 12344, 'vehicle_timestamp7'],
                ['B-212', 10, 12344, 'vehicle_timestamp8']
            ]
        )

        adder = TerminalModesAdder(
            locations_data,
            patterns_data,
            terminals_data
        )
        result = adder.fit_transform(vehicle_data).__array__().tolist()

        assert sorted(result) == [
            ['B-111', 12344, 'vehicle_timestamp3', '70059', False],
            ['B-111', 12345, 'vehicle_timestamp1', '70059', True],
            ['B-112', 12344, 'vehicle_timestamp4', '70059', False],
            ['B-112', 12345, 'vehicle_timestamp2', '70059', True],
            ['B-211', 12344, 'vehicle_timestamp7', '70038', True],
            ['B-211', 12345, 'vehicle_timestamp5', '70038', False],
            ['B-212', 12344, 'vehicle_timestamp8', '70038', True],
            ['B-212', 12345, 'vehicle_timestamp6', '70038', False]
        ]
