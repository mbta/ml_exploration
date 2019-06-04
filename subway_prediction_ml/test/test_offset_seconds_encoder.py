import os
from math import nan
import pandas as pd
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from offset_seconds_encoder import OffsetSecondsEncoder

class TestOffsetSecondsEncoder(unittest.TestCase):
    def test_transform_encodes_offset_seconds(self):
        test_data = pd.DataFrame(
            columns=['offset_departure_seconds_from_now'],
            data=[
                [nan],
                [-0.4],
                [37],
                [984],
                [1234]
            ]
        )
        result = OffsetSecondsEncoder(). \
            fit_transform(test_data). \
            __array__(). \
            tolist()
        assert(result == [
            ['nan'],
            ['negative'],
            ['0-500'],
            ['500-1000'],
            ['1000-1500']
        ])
