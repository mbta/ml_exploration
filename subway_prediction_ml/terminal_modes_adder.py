import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from location_id_translator import LocationIdTranslator


class TerminalModesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, locations_path, patterns_path, terminals_path):
        self.locations_path = locations_path
        self.patterns_path = patterns_path
        self.terminals_path = terminals_path

    def fit(self, vehicle_datapoints, y=None):
        return self

    def transform(self, vehicle_datapoints, y=None):
        terminals = self._gtfs_id_for_terminals()

        frame_with_terminals = pd.merge(
            vehicle_datapoints,
            right=terminals,
            left_on="pattern_id",
            right_on="pattern_id",
            how="inner"
        )
        frame_with_terminal_modes = pd.merge(
            frame_with_terminals,
            right=self._load_terminal_datapoints(),
            left_on=["generation", "terminal_gtfs_id"],
            right_on=["generation", "terminal_stop_id"],
            how="inner"
        ).drop(
            ["terminal_stop_id", "timestamp_y", "pattern_id"],
            axis=1
        ).rename({"timestamp_x": "timestamp"}, axis="columns")
        return frame_with_terminal_modes

    # Build a dataframe mapping each pattern ID to the corresponding terminal
    # stop GTFS ID
    def _gtfs_id_for_terminals(self):
        patterns = self._load_patterns()
        locations = LocationIdTranslator(self.locations_path)
        patterns["terminal_gtfs_id"] = patterns["terminal_stop"] \
            .apply(lambda loc_id: locations.get(loc_id))
        patterns = patterns.dropna().drop("terminal_stop", axis=1)
        return patterns

    # Build a dataframe mapping each pattern ID to the corresponding terminal
    # stop location ID
    def _load_patterns(self):
        dropped_frame = pd.read_csv(
            self.patterns_path, usecols=['pattern_id', 'terminal_stop']
        ).dropna()
        dropped_frame["terminal_stop"] = dropped_frame["terminal_stop"] \
            .apply(lambda str: str.split("|")[0])
        return dropped_frame

    # Build a dataframe with all logged terminal-mode datapoints
    def _load_terminal_datapoints(self):
        return pd.read_csv(
            self.terminals_path,
            usecols=[
                'automatic',
                'generation',
                'terminal_stop_id',
                'timestamp'
            ],
            dtype={"terminal_stop_id": str}
        )
