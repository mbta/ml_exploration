import pandas as pd
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from route_filter import RouteFilter


class TestRouteFilter(unittest.TestCase):
    def test_transform_removes_all_but_one_line(self):
        route_filter = RouteFilter('Q')
        data = pd.DataFrame(
            columns=['vehicle_id', 'other_value'],
            data=[
                ['G', 1234],
                ['Q', 5678],
                ['M', 9012],
                ['Q', 3456]
            ]
        )
        result = route_filter.transform(data).__array__().tolist()
        assert(result == [['Q', 5678], ['Q', 3456]])
