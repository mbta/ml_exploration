import functools
import pandas as pd

# Converts a location ID to the GTFS ID of the appropriate parent stop


class LocationIdTranslator:
    def __init__(self, locations_path):
        self._translation_dict = self._build_translation_dict(locations_path)

    def all_translated(self):
        return set(self._translation_dict.values())

    def get(self, loc_id):
        return self._translation_dict.get(loc_id)

    def _build_translation_dict(self, locations_path):
        dropped_frame = pd.read_csv(
            locations_path,
            usecols=['loc_id', 'gtfs_stop_id']
        )
        dropped_frame = dropped_frame.dropna()
        is_stop = dropped_frame.apply(lambda x: x != "0")
        stops_only = dropped_frame[is_stop].dropna()
        return functools.reduce(self._parent_stop, stops_only.__array__(), {})

    def _parent_stop(self, acc, x):
        location_id = x[0]
        gtfs_id = x[1]
        parent_stops = {
            "Alewife-01": "70061",
            "Alewife-02": "70061",
            "Braintree-01": "70105",
            "Braintree-02": "70105",
            "Forest Hills-01": "70001",
            "Forest Hills-02": "70001",
            "Oak Grove-01": "70036",
            "Oak Grove-02": "70036",
            "Government Center-Brattle": "70202"
        }
        parent_id = parent_stops.get(gtfs_id)
        if parent_id:
            gtfs_id = parent_id

        acc[location_id] = gtfs_id
        return acc
