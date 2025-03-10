"""**Linear Regression API**

This module is wrapper of the Scikit-learn, Scipy lsq_linear and custom linear regression algorithms.
"""

import platform
import traceback

import numpy as np
import pandas as pd
from formulae import design_matrices
from scipy.optimize import lsq_linear
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .pylme import LinearRegressionConstrained
from .utils import _process_formula


class LinearRegression(BaseEstimator):
    """
    A Linear Regression estimator with the option to use different backends for fitting the model.

    Parameters
    ----------
    backend : str, optional (default='sklearn')
        The backend to use for fitting the linear regression model. Available options are 'sklearn', 'scipy', and 'custom'.
    formula : str, required
        A string formula specifying the linear regression model to fit.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Estimated coefficients for the linear regression model.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()
    >>> lr = LinearRegression(formula='target ~ CRIM + ZN + INDUS + CHAS + NOX')
    >>> lr.fit(boston)
    >>> lr.predict(boston)
    """

    def __init__(
        self,
        formula=None,
        backend="sklearn",
    ):
        """
        Initialize the LinearRegression object.

        Parameters
        ----------
        backend : str, optional
            The backend to use for fitting the model, default "sklearn".
        formula : str
            The formula to be passed for fitting the model.

        Raises
        ------
        ValueError
            If the formula is not passed or is not in string format or if the backend is not "sklearn", "scipy", or "custom".
        """

        if formula == None:
            raise ValueError("Formula must be an input")
        elif not isinstance(formula, str):
            raise ValueError("Formula must be of type str")
        else:
            self.formula = formula

        # warnings.resetwarnings()
        if backend in ["sklearn", "scipy", "custom"]:
            self.backend = backend
        else:
            raise ValueError("Backend must be 'sklearn','scipy','custom'")

    def fit(
        self,
        df,
        box_constraints=None,
        **kwargs,
    ):
        """
        Fit the linear regression model to the input data.

        Parameters
        ----------
        df : pandas.DataFrame
            The input data to fit the model to.
        box_constraints : dict, optional
            A dictionary of variable names and their lower and upper bounds for box constraints on the model, default None.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the backend fitting function.

        Returns
        -------
        LinearRegression
            The fitted LinearRegression object.

        Raises
        ------
        ValueError
            If the backend is set to "sklearn" and box constraints are specified.
        """

        self.formula = _process_formula(self.formula, df.columns)
        self.dm = design_matrices(self.formula, df)

        y = np.array(self.dm.response)
        X = self.dm.common.design_matrix

        self.box_constraints = box_constraints

        if self.backend == "sklearn":
            if box_constraints:
                raise ValueError(
                    f"Backend options available for box_constraints are ['scipy','custom']"
                )
        else:
            box_constraints_all = dict.fromkeys(
                self.dm.common.terms.keys(), (-np.inf, np.inf)
            )
            if box_constraints is not None:
                for var in self.box_constraints.keys():
                    box_constraints_all[var] = self.box_constraints[var]

            self.lower_limits = list(list(zip(*list(box_constraints_all.values())))[0])
            self.upper_limits = list(list(zip(*list(box_constraints_all.values())))[1])

        try:
            if self.backend == "sklearn":
                self.model = linear_model.LinearRegression(fit_intercept=False)
                self.model.fit(X, y)
                self.coef_ = self.model.coef_
            elif self.backend == "scipy":
                self.model = None
                self.model = lsq_linear(
                    X, y, bounds=(self.lower_limits, self.upper_limits), **kwargs
                )
                self.coef_ = self.model.x
            elif self.backend == "custom":
                self.model = LinearRegressionConstrained(self.formula)
                self.model.fit(df, box_constraints=self.box_constraints, **kwargs)
                # self.model.intercept_ = self.model.beta[0]
                self.coef_ = self.model.beta

            self.is_fitted_ = True
        except:
            print(traceback.format_exc())
            self.is_fitted_ = False
        return self

    def get_coefficients(self):
        """
        Get the coefficients for the linear regression model.

        Returns
        -------
        dict
            A dictionary of variable names and their corresponding coefficient values.
        """

        check_is_fitted(self, "is_fitted_")
        fx_coef = {}
        # beta = self.beta.copy()
        # beta = [x if abs(x) > 1e-12 else 0 for x in beta]
        for term, slc in self.dm.common.slices.items():
            fx_coef[term] = self.coef_[slc][0]
        return fx_coef

    def predict(self, df):
        """
        Use the fitted linear regression model to make predictions on new data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to make predictions on.

        Returns
        -------
            The predicted values for the new data.
        """

        X = self.dm.common.evaluate_new_data(df).design_matrix

        check_is_fitted(self)
        _ = check_array(X.copy())
        if self.backend == "sklearn":
            return self.model.predict(X)
        elif self.backend == "scipy":
            return np.dot(X, self.model.x)
        if self.backend == "custom":
            return self.model.predict(df)
