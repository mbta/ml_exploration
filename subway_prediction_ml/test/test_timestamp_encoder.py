import os
import pandas as pd
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from timestamp_encoder import TimestampEncoder


class TestTimestampEncoder(unittest.TestCase):
    def test_transform_adds_day_of_week_and_binned_time_of_day(self):
        test_data = pd.DataFrame(
            columns=['timestamp'],
            data=[
                ['2019-06-02T15:58:00.000000Z'],
                ['2019-06-03T14:58:00.000000Z'],
                ['2019-06-03T15:58:00.000000Z']
            ]
        )

        result = TimestampEncoder(). \
            fit_transform(test_data). \
            __array__(). \
            tolist()
        assert(
            result == [
                ['2019-06-02T15:58:00.000000Z', 6, 63],
                ['2019-06-03T14:58:00.000000Z', 0, 59],
                ['2019-06-03T15:58:00.000000Z', 0, 63]
            ]
        )
