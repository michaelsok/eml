import numpy as np

from .ice import ICE


class PartialDependence(ICE):
    def interpret(self, X, feature, with_confidence=False, n_splits=20, how='quantile', **confidenceargs):
        abscissa, ordinates = super().interpret(X, feature=feature, n_splits=n_splits, how=how)
        if with_confidence:
            means, lower, upper = self._compute_confidence_on(ordinates, **confidenceargs)
            return abscissa, means, (lower, upper)
        return abscissa, ordinates.mean(axis=1)

    @staticmethod
    def _compute_confidence_on(array, coef=None, prior='gaussian'):
        means, std = np.array(array).mean(axis=1), np.array(array).std(axis=1)
        if (coef is None) and (prior == 'gaussian'):
            coef = 1.96
        return means, means - coef * std, means + coef * std
