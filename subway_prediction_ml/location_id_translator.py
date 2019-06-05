# Converts a location ID to the GTFS ID of the appropriate parent stop


class LocationIdTranslator:
    def __init__(self, locations_frame):
        self._translation_dict = self._build_translation_dict(locations_frame)

    def all_translated(self):
        return set(self._translation_dict.values())

    def get(self, loc_id):
        return self._translation_dict.get(loc_id)

    def _build_translation_dict(self, locations_frame):
        stops_only = locations_frame. \
            dropna(subset=['gtfs_stop_id']). \
            query("gtfs_stop_id != '0'")
        stops_only.loc[:, 'gtfs_stop_id'] = stops_only['gtfs_stop_id'].map(
            self._parent_stop
        )
        return dict(zip(stops_only.loc_id, stops_only.gtfs_stop_id))

    def _parent_stop(self, stop_id):
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
        return parent_stops.get(stop_id) or stop_id
