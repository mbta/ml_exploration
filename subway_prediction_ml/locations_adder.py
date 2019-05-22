import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin


class LocationsAdder(BaseEstimator, TransformerMixin):
    def __init__(self, locations_path):
        self.locations_path = locations_path

    def fit(self, vehicle_datapoints, y=None):
        return self

    def transform(self, vehicle_datapoints, y=None):
        n_hot_locations = self._n_hot_vehicle_locations_by_generation(
            vehicle_datapoints
        )

        generations = vehicle_datapoints["generation"].to_list()

        n_hot_per_row_list_of_lists = list(
            map(
                lambda generation: n_hot_locations[generation], generations
            )
        )
        n_hot_per_row = pd.DataFrame(
            n_hot_per_row_list_of_lists,
            columns=self._all_locs_sorted()
        )
        return vehicle_datapoints.join(n_hot_per_row, how="inner")

    # Return array of all location IDs in lexicographical order
    def _all_locs_sorted(self):
        loc_frame = pd.read_csv(self.locations_path)
        return sorted(loc_frame["loc_id"].__array__())

    # Builds a map of generations to the list of n-hot encoded vehicle
    # positions for that generation.
    def _n_hot_vehicle_locations_by_generation(self, vehicle_datapoints):
        grouped_frame = vehicle_datapoints.groupby(by="generation")

        occupied_locations_by_generation = {}
        generational_n_hot_locations = {}

        for generation in grouped_frame:
            generation_id = generation[0]
            locations = generation[1]["current_location_id"]

            occupied_locations_by_generation[generation_id] = set()
            for location in locations:
                occupied_locations_by_generation[generation_id].add(location)

            generational_n_hot_locations[generation_id] = defaultdict(
                lambda: 0
            )
            for location in occupied_locations_by_generation[generation_id]:
                generational_n_hot_locations[generation_id].update(
                    {location: 1}
                )

        generational_n_hot_location_lists = {}
        for generation in generational_n_hot_locations:
            location_map = generational_n_hot_locations[generation]
            n_hot_columns = []
            for location in self._all_locs_sorted():
                n_hot_columns.append(location_map[location])
                generational_n_hot_location_lists[generation] = n_hot_columns

        return generational_n_hot_location_lists