import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from formulae import design_matrices
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tabulate import tabulate


class StatsModelsLME(BaseEstimator):
    """
    Linear Mixed Effects model implemented using Statsmodels.

    Parameters:
    -----------
    formula : str
        Formula defining the model. Example: 'y ~ x1 + x2 + (1|group)'.
        Only one grouping factor is supported.

    """

    def __init__(self, formula):
        """
                Initialize StatsModelsLME object.

        Parameters
        ----------
        formula : str
            A string specifying the formula for the linear mixed effects model.
                Raises
                ------
                NotImplementedError
                        If Linear Mixed Effects not implemented with  proper formula type
        """

        if formula.split()[-1].strip() == ".":
            raise NotImplementedError(
                "Linear Mixed Effects not implemented with  above formula type"
            )
        self.formula = formula

    def fit(self, data, method="lbfgs", verbose=True):
        """
        Fit a linear mixed effects model to the data.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to fit the model to.

        method : str, optional
            The optimization method to use when fitting the model. default = "lbfgs".

        verbose : bool, optional
            Whether to print out the model coefficients after fitting. default = True.

        Returns
        -------
        self : object
            Returns self.
                Raises
                ------
                NotImplementedError
                        If Statsmodel LME is not implemented for multiple group factors
        """

        self.dm = design_matrices(self.formula, data)
        group_factor = []
        group_expr_data = []
        term_z_index = {}
        i = 0
        for key, gterm in self.dm.group.terms.items():
            # print(gterm)
            group_factor.append(gterm.factor.name)
            group_expr_data.append(
                self.dm.group.design_matrix[:, self.dm.group.slices[key]].sum(axis=1)
            )
            term_z_index[key] = i
            i += 1
        group_factor = list(set(group_factor))

        if len(group_factor) > 1:
            raise NotImplementedError(
                "Statsmodel LME is not implemented for multiple group factors"
            )
        else:
            group_factor = group_factor[0]

        X = self.dm.common.as_dataframe()
        re_cols = list(term_z_index.keys())
        Z = pd.DataFrame(np.array(group_expr_data).T, columns=re_cols)
        clusters = self.dm.data[group_factor].astype(str)
        y = np.array(self.dm.response)

        self.md = sm.MixedLM(endog=y, exog=X, groups=clusters, exog_re=Z)
        self.sme_fit = self.md.fit(method=method)

        self.is_fitted_ = True

        if verbose:
            self._print_coeff()

        return self

    def get_coefficients(self):
        """
        Get the coefficients for the linear mixed effects model.

        Returns
        -------
        dict
            A dictionary containing the coefficients for the fixed and random effects.
        """

        check_is_fitted(self, "is_fitted_")
        fx_coef = {}
        for term in self.dm.common.terms:
            fx_coef[term] = self.sme_fit.fe_params[term]

        rnd_coef = {}
        theta = self.sme_fit.random_effects
        for term, grp_item in self.dm.group.terms.items():
            # check
            rnd_coef[term] = {x: theta[x][term] for x in grp_item.groups}

        return {"common": fx_coef, "group": rnd_coef}

    def _print_coeff(self):
        """
        Prints out the fixed and random effects coefficients for the linear mixed effects model.
        """

        print("Fixed Effects Coefficients")
        fx_coef = [["Term", "Coeff"]]

        for term in self.dm.common.terms:
            fx_coef.append([term, self.sme_fit.fe_params[term]])

        print(
            tabulate(
                fx_coef,
                headers="firstrow",
                tablefmt="psql",
                floatfmt=".4f",
            )
        )

        print("Random Effects Coefficients")
        rnd_coef = [["Term", "Mean", "Var", "Std", "Min", "Max"]]
        theta = self.sme_fit.random_effects
        for term, grp_item in self.dm.group.terms.items():
            theta_term = np.array([theta[x][term] for x in grp_item.groups])
            rnd_coef.append(
                [
                    term,
                    theta_term.mean(),
                    theta_term.var(),
                    theta_term.std(),
                    theta_term.min(),
                    theta_term.max(),
                ]
            )
        print(
            tabulate(
                rnd_coef,
                headers="firstrow",
                tablefmt="psql",
                floatfmt=".4f",
            )
        )

    def predict(self, data):
        """
        Use the fitted linear mixed effects model to make predictions on new data.

        Parameters
        ----------
        df : pandas.DataFrame
            The data to make predictions on.

        Returns
        -------
        The predicted values for the new data.
        """

        check_is_fitted(self, "is_fitted_")

        final_coefs = self.get_coefficients()
        fixed_coef = np.array(list(final_coefs["common"].values()))
        x_fixed = self.dm.common.evaluate_new_data(data).design_matrix
        y_pred_fixed = np.dot(x_fixed, fixed_coef)

        rnd_coef = final_coefs["group"]
        y_pred_random = np.zeros_like(y_pred_fixed)
        group_feat_matrix = self.dm.group.evaluate_new_data(data).design_matrix
        for term, grp_item in self.dm.group.terms.items():
            matrix = group_feat_matrix[:, self.dm.group.slices[term]]
            coef = np.array(list(rnd_coef[term].values()))
            temp_random_pred = np.dot(matrix, coef)

            y_pred_random += temp_random_pred

        y_pred = y_pred_fixed + y_pred_random
        return y_pred
