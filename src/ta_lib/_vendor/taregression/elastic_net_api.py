"""**ElasticNet Regression API**

This module is wrapper of the Scikit-learn, glmnet-python and ta regression elasticnet algorithms.
"""

import platform
import traceback
import warnings

import numpy as np
import pandas as pd
from formulae import design_matrices
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .pylme import ElasticNetConstrained
from .utils import _check_bounds, _process_formula


class ElasticNet(BaseEstimator):
    """
    Elastic Net model with support for different backends including 'sklearn', 'glmnet' and 'custom'.

    Parameters
    ----------
    backend : str, optional
        The backend to use for Elastic Net regression. One of 'sklearn', 'glmnet', 'custom', by default 'sklearn'.
    formula : str, optional
        The formula to use for model fitting. Cannot be None.

    Attributes
    ----------
    model : object
        The fitted model object.
    coef_ : ndarray
        The estimated coefficients for the model.
    is_fitted_ : bool
        Whether the model has been fitted or not.

    Raises
    ------
    ValueError
        * If the formula is not passed or is not in string format or if the backend is not "sklearn", "glmnet", or "custom".
        * If OS is windows system while passing backend = "glmnet"
    """

    def __init__(self, formula=None, backend="sklearn"):
        """Initializes the ElasticNet object."""

        if formula == None:
            raise ValueError("Formula must be an input")
        elif not isinstance(formula, str):
            raise ValueError("Formula must be of type str")
        else:
            self.formula = formula

        if backend in ["sklearn", "glmnet", "custom"]:
            self.backend = backend
        else:
            raise ValueError("Backend must be 'sklearn','glmnet','custom'")

        if backend == "glmnet":
            os_type = platform.system()  # Linux, Darwin, Windows

            if os_type == "Windows":
                raise ValueError(
                    "glmnet package is not available for Windows. Try using WSL or Linux"
                )
            else:
                try:
                    import glmnet
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "Could not import glmnet package. Please install glmnet. `pip install glmnet`"
                    )

    def fit(
        self,
        df,
        box_constraints=None,
        alpha=0.5,
        lmbd=None,
        **kwargs,
    ):
        """
        Fits the elastic net model to the provided data.

        Parameters
        ----------
        df: pandas DataFrame
            The data to fit the algorithm to.
        box_constraints: dict, optional
            A dictionary of variable names and their lower and upper bounds for box constraints on the model, default None
        alpha: float, optional
            The alpha value for the Elastic Net model, default 0.5
        lmbd: float, optional
            The lambda value for the Elastic Net model, default None
        **kwargs: dict, optional
            Additional keyword arguments to pass to the backend fitting function.

        Returns
        -------
        self: ElasticNet
            The ElasticNet object with the model fitted to the provided data.

        Raises
        ------
        ValueError
            * If the selected backend is not one of the possible options for the given lower and upper bound constraints.
                        * If backend is sklearn and lambda value is not provided
        """

        self.box_constraints = box_constraints
        self.alpha = alpha
        self.lmbd = lmbd

        self.formula = _process_formula(self.formula, df.columns)
        self.dm = design_matrices(self.formula, df)

        y = np.array(self.dm.response)
        X = self.dm.common.design_matrix

        backend_options = {
            (0, 0): ["sklearn", "custom", "glmnet"],
            (1, 0): ["custom"],
            (1, 1): ["glmnet", "custom"],
        }  # statsmodels not implemeted yet

        if self.backend == "sklearn":
            if box_constraints:
                raise ValueError(
                    f"Backend options available for box_constraints are ['custom', 'glmnet']"
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

            bounds, bounds_with_zero = _check_bounds(
                self.lower_limits, self.upper_limits
            )

            backend_options = backend_options[(bounds, bounds_with_zero)]

            if self.backend not in backend_options:
                raise ValueError(
                    f"Backend options available for given bounds are {backend_options}"
                )
        try:
            if self.backend == "sklearn":
                if not self.lmbd:
                    raise ValueError(
                        " for backend 'sklearn', you must pass lambda value "
                    )
                self.model = linear_model.ElasticNet(
                    alpha=self.lmbd, l1_ratio=self.alpha, fit_intercept=False, **kwargs
                )
                self.model.fit(X, y)
                self.coef_ = self.model.coef_
            elif self.backend == "glmnet":
                if self.lmbd:
                    warnings.warn("'lmbd' parameter will not be considered for glmnet.")
                import glmnet

                self.model = glmnet.ElasticNet(
                    alpha=self.alpha,
                    lower_limits=self.lower_limits,
                    upper_limits=self.upper_limits,
                    fit_intercept=False,
                    **kwargs,
                )
                self.model.fit(X, y)
                self.coef_ = self.model.coef_
                self.lmbd = self.model.lambda_best_
            elif self.backend == "custom":

                self.model = ElasticNetConstrained(
                    self.formula, alpha=self.alpha, lmbd=self.lmbd
                )
                use_coord_descent = kwargs.pop("use_coord_descent", None)
                if not use_coord_descent:
                    if self.lmbd:
                        use_coord_descent = False
                    else:
                        use_coord_descent = True
                elif use_coord_descent:
                    if self.lmbd:
                        warning.warn(
                            "As input use_coord_descent = True, lmbd value will not be considered"
                        )

                self.model.fit(
                    df,
                    box_constraints=self.box_constraints,
                    use_coord_descent=use_coord_descent,
                    **kwargs,
                )
                self.coef_ = np.array(self.model.beta)
                if use_coord_descent:
                    self.lmbd_ = self.model.best_lmbd

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

    def predict(self, df, lmbd=None):
        """
        Use the fitted elastic net model to make predictions on new data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to make predictions on.
        lmbd : float
            To be used when backend = "glmnet", default None.

        Returns
        -------
        The predicted values for the new data.
        """

        # pd.options.mode.chained_assignment = None
        check_is_fitted(self, "is_fitted_")

        X = self.dm.common.evaluate_new_data(df).design_matrix

        check_is_fitted(self)
        _ = check_array(X.copy())
        if self.backend == "sklearn":
            return self.model.predict(X)
        elif self.backend == "custom":
            return self.model.predict(df)
        elif self.backend == "glmnet":
            return self.model.predict(X, lamb=lmbd)
