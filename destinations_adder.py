import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from location_id_translator import LocationIdTranslator


# Build a dataframe with every combination of given vehicle datapoints
# and possible destination.
#
# Note that we don't currently check that the destination makes sense, for
# example, there will be rows for the Blue Line with a destination of
# Kendall. Currently these get eliminated in a later step. It might be
# more efficient to do so up-front here, though we'd then have to have this
# code know somehow what stops are on what lines.

class DestinationsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, locations_path):
        self.locations_path = locations_path

    def fit(self, X, y=None):
        return self

    def transform(self, vehicle_datapoints, y=None):
        gtfs_ids = LocationIdTranslator(self.locations_path).all_translated()
        blank_frame = pd.DataFrame()
        new_frames = map(
            lambda gtfs_id: self._vehicle_datapoints_with_destination_gtfs_id(
                vehicle_datapoints,
                gtfs_id
            ),
            gtfs_ids
        )
        return blank_frame.append(list(new_frames))

    def _vehicle_datapoints_with_destination_gtfs_id(
        self,
        vehicle_datapoints,
        gtfs_id
    ):
        new_frame = vehicle_datapoints.copy()
        new_frame["destination_gtfs_id"] = str(gtfs_id)
        return new_frame
