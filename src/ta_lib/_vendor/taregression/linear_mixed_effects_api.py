"""**Linear Mixed Effects Regression API**

This module is wrapper of the statsmodel linear minxed effects and custom linear mixed effects algorithms.
"""

import platform
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_array, check_is_fitted

from .pylme import LinearMixedEffectsConstrained
from .statmodels_lme import StatsModelsLME

os_type = platform.system()  # Linux, Darwin, Windows


class LinearMixedEffects(LinearModel):
    """
    A class for fitting linear mixed-effects models with optional box constraints.

    Parameters
    ----------
    backend : str, default = "statsmodels"
        The implementation of linear mixed-effects model to use. Available options are
        "statsmodels" and "custom".
    formula : str
        The formula for the model.

    Attributes
    ----------
    formula : str
        The formula for the model.
    backend : str
        The implementation of linear mixed-effects model to use.
    box_constraints : list of tuples, optional
        The box constraints on the model parameters.

    """

    def __init__(self, formula=None, backend="statsmodels"):
        """
        Initialize LinearMixedEffects object.

        Parameters
        ----------
        backend : str, default='statsmodels'
            The backend used for fitting the model. Can be 'statsmodels' or 'custom'.
        formula : str
            The formula for the linear mixed effects model.

        Raises
        ------
        ValueError
            If the formula is not passed or is not in string format or if the backend is not "statsmodels", or "custom".
        """

        if formula == None:
            raise ValueError("Formula must be an input")
        elif not isinstance(formula, str):
            raise ValueError("Formula must be of type str")
        else:
            self.formula = formula

        if backend in ["statsmodels", "custom"]:
            self.backend = backend
        else:
            raise ValueError("reg_type must be 'statsmodels','custom'")

    def fit(
        self,
        df,
        box_constraints=None,
        **kwargs,
    ):
        """
        Fit the linear mixed effects model with optional box constraints.

        Parameters
        ----------
        df : pandas DataFrame
            The input data for fitting the model.
        box_constraints : dict, optional
            The box constraints used for fitting the model. Only available for the 'custom' backend.
        **kwargs
            Other parameters to be passed to the model fitting function.

        Returns
        -------
        self : LinearMixedEffects
            Fitted estimator.

        Raises
        ------
        ValueError
            * If backend is 'statsmodels' and box_constraints is not None.
            * If a backend option is passed other than 'statsmodels', 'custom'
            * If an exception occurs during model fitting.
        """

        self.box_constraints = box_constraints
        backend_options = ["statsmodels", "custom"]
        if self.backend == "statsmodels":
            if box_constraints:
                raise ValueError(
                    f"Backend options available for box_constraints are 'custom'"
                )

        if self.backend not in backend_options:
            raise ValueError(
                f"Backend options available for Linear Mixed Effects and given bounds are {backend_options}"
            )

        try:
            if self.backend == "statsmodels":
                self.model = StatsModelsLME(self.formula)
                self.model.fit(df, **kwargs)
            elif self.backend == "custom":
                self.model = LinearMixedEffectsConstrained(self.formula)
                self.model.fit(df, box_constraints=self.box_constraints, **kwargs)

            self.is_fitted_ = True
        except:
            print(traceback.format_exc())
            self.is_fitted_ = False

        return self

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

        check_is_fitted(self, "is_fitted_")
        # _ = check_array(X.copy())
        return self.model.predict(df)

    def get_coefficients(self):
        """
        Get the coefficients for the linear mixed effects model.

        Returns
        -------
        dict
            A dictionary of variable names and their corresponding coefficient values.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """

        check_is_fitted(self, "is_fitted_")

        return self.model.get_coefficients()
