import numpy as np
from sklearn.metrics import mean_squared_error


class ModelEvaluator:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels
        self.predictions = self.model.predict(self.features)
        self._binned_accuracy = None
        self._rmse = None

    def binned_accuracy(self):
        self._binned_accuracy = self._binned_accuracy or \
            self._calculate_binned_accuracy()
        return self._binned_accuracy

    def rmse(self):
        self._rmse = self._rmse or self._calculate_rmse()
        return self._rmse

    def _calculate_binned_accuracy(self):
        min_accuracy_bounds = map(self._min_accuracy_bounds, self.labels)
        max_accuracy_bounds = map(self._max_accuracy_bounds, self.labels)
        errors = self.predictions - self.labels
        self._in_bounds = [
            error >= min and error <= max
            for (error, min, max) in zip(
                errors,
                min_accuracy_bounds,
                max_accuracy_bounds
            )
        ]
        n_within_bounds = len(list(filter(lambda x: x, self._in_bounds))) * 1.0
        return n_within_bounds / len(self._in_bounds)

    def _calculate_rmse(self):
        mse = mean_squared_error(self.labels, self.predictions)
        return np.sqrt(mse)

    def _max_accuracy_bounds(self, seconds):
        if seconds < 180:
            return 60
        elif seconds < 360:
            return 120
        elif seconds < 720:
            return 210
        else:
            return 360

    def _min_accuracy_bounds(self, seconds):
        if seconds < 180:
            return -60
        elif seconds < 360:
            return -90
        elif seconds < 720:
            return -150
        else:
            return -240
