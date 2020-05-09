from .ice import IndividualConditionalExpectation


class PartialDependence(IndividualConditionalExpectation):
    def interpret(self, X, feature, n_splits=20, how='quantile'):
        abscissa, ordinates = super().interpret(X, feature=feature, n_splits=n_splits, how=how)
        return abscissa, ordinates.mean(axis=1)
