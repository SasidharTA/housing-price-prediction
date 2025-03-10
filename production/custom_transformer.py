from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that demonstrates both forward and inverse transforms.

    This transformer scales the input data by a factor and also supports
    the inverse transformation to restore the data to its original form.

    Parameters
    ----------
    factor : float, default=1.0
        The factor by which to scale the input data during the transform step.
    """

    def __init__(self, factor=1.0):
        self.factor = factor

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this example).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to fit on.
        y : array-like, shape (n_samples,), optional, default=None
            The target values (not used in this example).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Scale the input data by the specified factor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data.
        """
        return X * self.factor

    def inverse_transform(self, X):
        """
        Inverse the scaling by dividing by the specified factor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The transformed data to revert.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
            The original data before scaling.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise ValueError("Input must be an array-like object.")
        return X / self.factor