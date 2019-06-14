import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Chunker(BaseEstimator):
    def __init__(self, estimator, chunk_size, **estimator_kwargs):
        self.estimator = estimator(**estimator_kwargs)
        self.chunk_size = chunk_size

    def fit(self, X, y):
        X_chunks = np.array_split(X, self.chunk_size)
        y_chunks = np.array_split(y, self.chunk_size)
        for X_chunk, y_chunk in zip(X_chunks, y_chunks):
            self.estimator.partial_fit(X_chunk, y_chunk)
        return self

    def predict(self, X):
        X_chunks = np.array_split(X, self.chunk_size)
        result_chunks = list(map(self.estimator.predict, X_chunks))
        return np.concatenate(result_chunks)
