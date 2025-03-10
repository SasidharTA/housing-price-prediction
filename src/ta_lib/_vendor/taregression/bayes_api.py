"""**Bayesian Regression API**

This module is wrapper of the Tensorflow Probability and Bambi bayesian regression libraries.
"""

import platform
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model._base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from .bambi_bayesian import BambiBayesian
from .tfp_bayesian import TFPBayesian

os_type = platform.system()  # Linux, Darwin, Windows


class BayesianRegression(BaseEstimator):
    """
    A class for fitting Bayesian regression models with optional priors.

    Parameters
    ----------
    backend : str, default="bambi"
        The implementation of Hierarchical Bayesian model to use. Available options are
        "bambi" and "tfp".
    formula : str
        The formula for the model.

    Attributes
    ----------
    formula : str
        The formula for the model.
    backend : str
        The implementation of Hierarchical Bayesian  model to use.

    Raises
    ------
    ValueError
        If the formula is not passed or is not in string format or if the backend is not "tfp", or "bambi".
    """

    def __init__(self, formula=None, backend="bambi"):
        """Initialize Bayesian object."""

        if formula == None:
            raise ValueError("Formula must be an input")
        elif not isinstance(formula, str):
            raise ValueError("Formula must be of type str")
        else:
            self.formula = formula

        if backend in ["tfp", "bambi"]:
            self.backend = backend
        else:
            raise ValueError("backend must be 'tfp','bambi'")

    def fit(
        self,
        df,
        priors=None,
        bijectors=None,
        model_params=None,
    ):
        """
        Fit the Bayesian model with optional priors and bijectors.

        Parameters
        ----------
        df : pandas DataFrame
            The input data for fitting the model.

        priors : dict, optional
            The priors used for fixed and random effects estimates.
            If backend is 'tfp' priors dictionary should be in a format given below

            * For fixed effects:

                .. code-block::

                    'fixed_feature': {'dist': distribution, 'mu': mu,'sd': sd}

            * For random effects:

                .. code-block::

                    'random_feature|group': {'mu': {'dist': distribution, 'mu': mu, 'sd': sd},
                                            'sd': {'dist': distribution, 'mu': mu, 'sd': sd}}

            If backend is 'bambi' priors dictionary is defined based on parameters of a 'PyMC' supported distribution

        bijectors: dict, optional
            The dictionary of variable and it's bijectors tyepe. This option is only available for 'tfp' backend.
            For fixed effects - 'fixed_feature': 'Identity'/'Exp'
            For random effects -'random_feature|group': {'mu':'Identity'/'Exp' ,'sd':'Identity'/'Exp'}

        model_params: dict, optional
             If backend is 'tfp' dictionary of model_params is in given format with default values

             .. code-block::

                {
                    'objective': 'regression',
                    'sampler':'nuts',
                    'num_chains':4,
                    'num_results':2000,
                    'num_burnin_steps':1000,
                    'num_leapfrog_steps':10,
                    'hmc_step_size':0.01,
                    'num_steps_between_results':2,
                    'max_energy_diff':1000,
                    'max_tree_depth':10,
                    'unrolled_leapfrog_steps':1,
                    'parallel_iterations':10
                }

            If backend is 'bambi' arguments to bambi.fit() functions are passed as dictionary.

        Returns
        -------
        self : BayesianModel
            Fitted estimator.

        Raises
        ------
        ValueError
            * If backend is 'bambi' and bijectors is not empty dictionary.
            * If a backend option is passed other than 'tfp', 'bambi'
            * If an exception occurs during model fitting.
        """

        if priors:
            self.priors = priors
        else:
            self.priors = {}

        backend_options = ["tfp", "bambi"]
        if self.backend == "bambi":
            if bijectors:
                raise ValueError(f"Backend options available for bijectors are 'tfp'")
        if bijectors:
            self.bijectors = bijectors
        else:
            self.bijectors = {}

        if self.backend not in backend_options:
            raise ValueError(
                f"Backend options available for Bayesian modeling and given bounds are {backend_options}"
            )

        if model_params:
            self.model_params = model_params
        else:
            self.model_params = {}

        try:
            if self.backend == "tfp":
                self.model = TFPBayesian(self.formula, self.model_params)
                self.model_result = self.model.fit(
                    data=df, priors=self.priors, bijectors=self.bijectors
                )
            elif self.backend == "bambi":
                self.model = BambiBayesian(self.formula, self.model_params)
                self.model_result = self.model.fit(data=df, priors=self.priors)
            self.is_fitted_ = True
        except:
            print(traceback.format_exc())
            self.is_fitted_ = False

        return self.model_result

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

    def get_summary(self):
        """
        Get the summary for the Bayesian Model

        Returns
        -------
        dict
            A dataframe with estimates of fixed and random effects

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """

        check_is_fitted(self, "is_fitted_")

        return self.model.get_summary()
