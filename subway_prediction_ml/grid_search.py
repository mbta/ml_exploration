from model_evaluator import ModelEvaluator

# Like sklearn's built-in GridSearchCV, but uses a separate training and
# validation set as opposed to doing cross-validation.


class GridSearch:
    def __init__(
        self,
        estimator,
        param_grid,
        validation_features,
        validation_labels,
        **estimator_kwargs
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.validation_features = validation_features
        self.validation_labels = validation_labels
        self.estimator_kwargs = estimator_kwargs

    def fit(self, X, y=None):
        self.scores_ = {}
        for grid_params in self.param_grid:
            estimator_args = {}
            estimator_args.update(self.estimator_kwargs)
            estimator_args.update(grid_params)
            model = self.estimator(**estimator_args)
            model.fit(X, y)
            evaluator = ModelEvaluator(
                model, self.validation_features, self.validation_labels
            )
            score = evaluator.binned_accuracy()
            self.scores_[model] = score
