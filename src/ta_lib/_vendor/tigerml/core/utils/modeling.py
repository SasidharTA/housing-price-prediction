from sklearn.exceptions import NotFittedError


def is_fitted(model, data_point):
    # import pdb
    # pdb.set_trace()
    if model.__module__.startswith("sklearn"):
        try:
            model.predict(data_point)
            return True
        except NotFittedError:
            return False
    else:
        return None


class Algo:
    """Algo class."""

    classification = "classification"
    regression = "regression"

    def is_classification(self, algo: str):
        """Algo class."""
        return algo == self.classification

    def is_regression(self, algo: str):
        """Algo class."""
        return algo == self.regression
