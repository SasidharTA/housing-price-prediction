import copy

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from formulae import design_matrices
from sklearn.base import BaseEstimator


class BambiBayesian(BaseEstimator):
    def __init__(self, formula, model_params={}):
        if formula.split()[-1].strip() == ".":
            raise NotImplementedError(
                "Bambi Bayesian not implemented with  above formula type"
            )

        self.model_params = model_params
        self.formula = formula

    def convert_to_bambi_priors(self, priors_dict_original):

        priors_dict = copy.deepcopy(priors_dict_original)
        bmb_priors = {}
        for k, v in priors_dict.items():
            sub_prior = {}
            dist = v.pop("dist")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    sub_prior[kk] = bmb.Prior(vv.pop("dist"), **vv)

                else:
                    sub_prior[kk] = vv
            bmb_priors[k] = bmb.Prior(dist, **sub_prior)
        return bmb_priors

    def fit(self, data, priors={}):

        import jax
        import tensorflow_probability.substrates.jax as tfp

        jax.scipy.special.erfcx = tfp.math.erfcx
        jax.scipy.special.erfcinv = tfp.math.erfcinv

        self.bmb_priors = self.convert_to_bambi_priors(priors)
        self.model = bmb.Model(formula=self.formula, data=data, priors=self.bmb_priors)
        self.model_result = self.model.fit(**self.model_params)
        return self.model_result

    def predict(self, df):
        """
        Use the fitted bambi model to make predictions on new data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to make predictions on.

        Returns
        -------
        The predicted values for the new data.
        """

        posterior = self.model_result.posterior.stack(sample=["chain", "draw"])
        dm = self.model._design
        predictions = []

        if dm.common != None:
            X = dm.common.evaluate_new_data(df).design_matrix
            beta = np.vstack([posterior[name] for name in list(dm.common.terms.keys())])
            predictions.append(np.dot(X, beta))

        if dm.group != None:
            Z = dm.group.evaluate_new_data(df).design_matrix
            mu = np.vstack([posterior[name] for name in list(dm.group.terms.keys())])
            predictions.append(np.dot(Z, mu))

        predictions = sum(predictions)
        return predictions.mean(axis=1)

    def get_summary(self):
        return az.summary(self.model_result)
