"""
TA Regression Constrained implimentations

this module contains constrained implimentation of below algorithms

* Linear Mixed effects 
* Elastic Net
* Linear Regression

"""

import traceback
import warnings

import nlopt
import numpy as np
import pandas as pd
from formulae import design_matrices
from numba import jit
from scipy.sparse import bsr_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tabulate import tabulate


class LinearMixedEffectsConstrained(BaseEstimator):
    """Linear mixed effects regression model with constraints on the estimated parameters.

    This class fits a linear mixed effects regression model to the input data. The class can
    handle fixed and random effects in the model. It uses non-linear optimization to estimate
    the parameters of the model. The class supports constraints on the estimated parameters.

    Parameters:
    -----------
    formula : str
        A formula string in the format "y ~ x1 + x2 + ...", where y is the response variable,
        and x1, x2, ... are the predictor variables.

    Attributes:
    -----------
    beta : numpy.ndarray
        An array of estimated fixed effect coefficients.
    theta : numpy.ndarray
        An array of estimated random effect coefficients.
    factors_train : dict
        A dictionary of factor levels for each random effect term.
    is_fitted_ : bool
        A boolean flag indicating whether the model has been fitted.
    """

    def __init__(self, formula):
        """
        Initializes the LinearMixedEffectsConstrained object with the given formula.

        Parameters
        ----------
        formula : str
        A string representing the formula for the linear mixed effects model.

        """
        # self.alpha = alpha
        # self.l1_ratio = l1_ratio
        self.formula = formula

    def fit(self, data, box_constraints=None, max_iter_opt=None, verbose=True):
        """Fits the linear mixed effects model to the given data.

            Parameters
            ----------
            data: pandas.DataFrame
        The input data to fit the model to.
            box_constraints: dict, optional
        A dictionary of variable names and their lower and upper bounds for box constraints on the model, default None.
            max_iter_opt: int, optional
                    An integer specifying the maximum number of iterations for the optimization algorithm.
            verbose: bool
                    A boolean flag indicating whether to print progress messages during the fitting process.

            Returns
            -------
            LinearMixedEffectsConstrained
        The fitted LinearMixedEffectsConstrained object.

        """

        self.train_data = data
        n_samples = len(data)
        # data parsing using formulae
        self.dm = design_matrices(self.formula, data)

        self.y = self.dm.response.as_dataframe().values
        self.x_fixed = self.dm.common.design_matrix
        try:
            x_random = self.dm.group.design_matrix
            self.x_random_sparse = bsr_matrix(x_random, shape=x_random.shape)
            self.random_eff = True
        except LookupError:
            if verbose:
                print("No Random effects")
            self.random_eff = False

        self._one_by_n = 1.0 / (self.x_fixed.shape[0])
        self._X = np.concatenate([self.x_fixed, self.x_random_sparse.todense()], axis=1)

        # parameter lengths
        self.beta_len = self.x_fixed.shape[1]
        if self.random_eff:
            theta_len = self.x_random_sparse.shape[1]
            self.param_len = self.beta_len + theta_len
            x0_fixed = np.random.rand(self.beta_len)
            x0_random = np.random.rand(theta_len)
        else:
            x0_fixed = np.zeros(self.beta_len)
            self.param_len = self.beta_len

        # checking the box_constraints
        fixed_terms = list(self.dm.common.terms.keys())

        fixed_upper_limits = np.ones(self.beta_len) * np.inf
        fixed_lower_limits = np.ones(self.beta_len) * -np.inf

        if self.random_eff:
            random_terms = list(self.dm.group.terms.keys())
            random_upper_limits = np.ones(theta_len) * np.inf
            random_lower_limits = np.ones(theta_len) * -np.inf
        else:
            random_terms = []
            random_upper_limits = []
            random_lower_limits = []
        if verbose:
            print("Fixed terms: ", fixed_terms)
            print("Random terms: ", random_terms)
        if box_constraints:
            for term, limits in box_constraints.items():
                limits = np.array(limits)
                # TODO: raise if _x0 is inf case
                if len(limits[~np.isnan(limits) & ~np.isinf(limits)]) == 1:

                    if np.isinf(limits.min()):
                        ll, ul = limits.max() / 10, limits.max()
                    elif np.isinf(limits.max()):
                        ll, ul = limits.min(), limits.min() * 10
                else:
                    ll, ul = limits.min(), limits.max()
                if term in fixed_terms:
                    slicer = self.dm.common.slices[term]
                    fixed_upper_limits[slicer] = ul
                    fixed_lower_limits[slicer] = ll
                    x0_fixed[slicer] = np.random.uniform(
                        low=ll, high=ul, size=len(x0_fixed[slicer])
                    )
                elif term in random_terms:
                    slicer = self.dm.group.slices[term]
                    random_upper_limits[slicer] = ul
                    random_lower_limits[slicer] = ll
                    x0_random[slicer] = np.random.uniform(
                        low=ll, high=ul, size=len(x0_random[slicer])
                    )
                else:
                    print(traceback.format_exc())
                    raise ValueError(
                        f' "{term}" is not among the terms'
                        + f"Fixed : {fixed_terms},"
                        + f"or Random : {random_terms}"
                    )

        # optimization

        opt = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, self.param_len)
        if box_constraints:
            opt.set_lower_bounds(list(fixed_lower_limits) + list(random_lower_limits))
            opt.set_upper_bounds(list(fixed_upper_limits) + list(random_upper_limits))

        opt.set_min_objective(self._objective)
        # opt.set_xtol_rel(1e-8)
        if max_iter_opt is None:
            # this way of calculating max iterations is not optimum
            max_iter_opt = 5000  # int(max(5000, (n_samples * self.param_len) / 3))
        print("max iter: ", max_iter_opt)
        opt.set_maxeval(max_iter_opt)
        if self.random_eff:
            x_soln = opt.optimize(list(x0_fixed) + list(x0_random))
            self.beta, self.theta = (
                x_soln[: self.beta_len],
                x_soln[self.beta_len :],
            )
        else:
            x_soln = opt.optimize(x0_fixed)
            self.beta, self.theta = (x_soln, None)
        print("Optimization run complete")
        print("number of function evals: ", opt.get_numevals())
        err_msg = opt.get_errmsg()
        if err_msg:
            print(err_msg)

        self.is_fitted_ = True

        if verbose:
            self._print_coeff()

        # factors
        if self.random_eff:
            self.factors_train = {}
            for grp_term in self.dm.group.terms.values():
                if grp_term.factor.name not in self.factors_train:
                    self.factors_train[grp_term.factor.name] = grp_term.groups

        return self

    def get_coefficients(self):
        """
        Retrieves the coefficients of the fitted linear mixed effects model.

        Returns
        -------
        dict
                A dictionary containing the fixed effects coefficients of the fitted model. The keys are the names of the fixed effects, and the values are the corresponding coefficients.
        """

        check_is_fitted(self, "is_fitted_")
        fx_coef = {}
        for term, slc in self.dm.common.slices.items():
            # slc = values["cols"]
            fx_coef[term] = self.beta[slc][0]
        if self.random_eff:
            rnd_coef = {}
            for term, grp_item in self.dm.group.terms.items():
                slc = self.dm.group.slices[term]
                rnd_coef[term] = {
                    x: y for x, y in zip(grp_item.groups, self.theta[slc])
                }

        return {"common": fx_coef, "group": rnd_coef}

    def _print_coeff(self):
        print("Fixed Effects Coefficients")
        fx_coef = [["Term", "Coeff"]]
        for term, slc in self.dm.common.slices.items():
            # slc = values["cols"]
            fx_coef.append([term, self.beta[slc][0]])
        print(
            tabulate(
                fx_coef,
                headers="firstrow",
                tablefmt="psql",
                floatfmt=".4f",
            )
        )
        if self.random_eff:
            print("Random Effects Coefficients")
            rnd_coef = [["Term", "Mean", "Var", "Std", "Min", "Max"]]
            for term, slc in self.dm.group.slices.items():
                # slc = values["cols"]
                theta_term = self.theta[slc]

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

    def _eq(self, x_fixed, x_random_sparse, beta, theta):
        return np.dot(x_fixed, beta) + x_random_sparse.dot(theta)

    def _objective(self, *args):

        beta, theta = args[0][: self.beta_len], args[0][self.beta_len :]
        y_pred = self._eq(self.x_fixed, self.x_random_sparse, beta, theta)

        y_diff = self.y.reshape(1, -1)[0] - y_pred.reshape(1, -1)[0]

        loss = np.dot(y_diff, y_diff.T)
        if args[1].size > 0:
            args[1][:] = -(self._one_by_n * np.dot(self._X.T, y_diff))
        return loss

    def score(self, *args):
        """
        Method used to calculate the R-squared score for the mixed effect model.
        It takes a variable-length argument tuple `args`, but this is ignored.

        Parameters:
        -----------
        args : tuple
                A variable-length argument tuple. This is ignored.

        Returns:
        --------
        score : float
                The R-squared score for the mixed effect model.
        """

        return r2_score(self.y, self.predict(self.train_data))

    def predict(self, data, verbose=True):
        """
        Use the fitted mixed effects model to make predictions on new data.

        Parameters:
        -----------
        data : pandas.DataFrame
                A pandas DataFrame with the same columns as the training data.
        verbose : bool
                A boolean parameter that determines whether to print warnings about unseen
                categorical values in the new data. Default is True.

        Returns:
        --------
        y_pred_random + y_pred_fixed : numpy.ndarray
                A numpy array of predicted response variable values for the new data.
        """
        pd.options.mode.chained_assignment = None
        check_is_fitted(self, "is_fitted_")

        x_fixed = self.dm.common.evaluate_new_data(data).design_matrix
        y_pred_fixed = np.dot(x_fixed, self.beta)

        y_pred_random = np.zeros_like(y_pred_fixed)
        data["Intercept"] = 1
        if self.random_eff:
            grp_coeffs = self.get_coefficients()["group"]
            for term in self.dm.group.terms:
                term_info = self.dm.group.terms[term]
                formula_term = term
                expr_name = term_info.expr.name
                factor_name = term_info.factor.name
                # print(formula_term, expr_name, factor_name)

                if term_info.kind in ["numeric", "intercept"]:
                    term_groups = list(data[factor_name].astype("str").unique())

                    diff = list(set(term_groups) - set(term_info.groups))
                    if len(diff) > 0:
                        if verbose:
                            print(
                                f"Term formula {formula_term},"
                                + f"factor {factor_name} has {len(diff)}"
                                + f"unseen values in the test data, those are \n {diff}"
                            )

                        _seen_data = data[data[factor_name].isin(term_info.groups)]
                        _unseen_data = data[~data[factor_name].isin(term_info.groups)]
                        _unseen_data.loc[:, "y_pred"] = (
                            term_info.expr.eval_new_data(_unseen_data)
                            * self.theta[self.dm.group.slices[formula_term]].mean()
                        )

                        if _seen_data.empty:

                            term_ypred = _unseen_data[["y_pred"]]

                        else:

                            x_random_seen = self.dm.group.evaluate_new_data(
                                _seen_data
                            ).design_matrix
                            _seen_data.loc[:, "y_pred"] = np.dot(
                                x_random_seen[:, self.dm.group.slices[formula_term]],
                                self.theta[self.dm.group.slices[formula_term]],
                            )

                            term_ypred = pd.concat(
                                [_seen_data[["y_pred"]], _unseen_data[["y_pred"]]]
                            )

                        term_ypred = term_ypred.sort_index()
                        y_pred_random += term_ypred["y_pred"]
                    else:

                        x_random_seen = self.dm.group.evaluate_new_data(
                            data
                        ).design_matrix
                        y_pred_random += np.dot(
                            x_random_seen[:, self.dm.group.slices[formula_term]],
                            self.theta[self.dm.group.slices[formula_term]],
                        )
                else:
                    raise NotImplementedError(
                        f"{type} of random effect is not implemented"
                    )

        return y_pred_random + y_pred_fixed


class ElasticNetConstrained(BaseEstimator):
    """Elastic Net Regression model with constraints.

    This class implements the Elastic Net Regression model with constraints.
    The Elastic Net model is a combination of Ridge and Lasso regression models,
    and is suitable for high-dimensional data. This implementation allows
    constraints to be imposed on the model coefficients through the `box_constraints`
    argument. The optimization algorithm used can be selected using the `nlopt_optim`
    argument.

    Parameters
    ----------
    formula : str
        A Patsy formula specifying the model.
    alpha : float
        Constant that multiplies the L1 penalty term. `alpha=0` is equivalent
        to a Ridge regression, while `alpha=1` is equivalent to a Lasso regression.
    lmbd : float, optional
        The regularization parameter. If not specified, `lmbd` will be estimated
        using cross-validation.
    box_constraints : dict, optional
        A dictionary specifying the lower and upper bounds for the coefficients
        associated with the formula terms.
    scale_data : bool, optional
        If `True`, scales the data to have zero mean and unit variance.
    nlopt_optim : int, optional
        The optimization algorithm used to minimize the objective function.
        Defaults to `nlopt.LN_BOBYQA`.
    use_coord_descent : bool, optional
        If `True`, uses coordinate descent instead of optimization. Defaults to `False`.
    verbose : bool, optional
        If `True`, prints progress messages.
    max_iter_opt : int, optional
        The maximum number of iterations allowed for the optimization algorithm.

    Attributes
    ----------
    dm : patsy.design_info.DesignMatrix
        The design matrix.
    beta : numpy.ndarray
        The estimated coefficients.
    best_lmbd : float
        The best value for the regularization parameter, found through cross-validation.
    lmbd_history : list
        A list containing the regularization parameter values tried during cross-validation.
    """

    def __init__(self, formula, alpha, lmbd=None):
        self.alpha = alpha
        self.formula = formula
        self.lmbd = lmbd

    def fit(
        self,
        data,
        box_constraints=None,
        scale_data=False,
        nlopt_optim=None,
        use_coord_descent=False,
        verbose=False,
        max_iter_opt=5000,
    ):
        """
        Fits the model to the input data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with independent and dependent variables.
        box_constraints : Dict[str, Tuple[Optional[float], Optional[float]]], optional
            A dictionary of constraints for the model coefficients. The keys of
            the dictionary should be the names of the variables and the values
            should be tuples with the lower and upper bounds for the
            corresponding variable. Defaults to None.
        scale_data : bool, optional
            Whether to standardize the data. Defaults to False.
        nlopt_optim : int, optional
            The optimization algorithm to use. If None, defaults to nlopt.LN_BOBYQA.
            See the documentation of the nlopt package for a full list of options.
            Defaults to None.
        use_coord_descent : bool, optional
            Whether to use coordinate descent for optimization. If True,
            `nlopt_optim` and `self.lmbd` parameters will be ignored.
            Defaults to False.
        verbose : bool, optional
            Whether to print additional information during optimization.
            Defaults to False.
                max_iter_opt: int, optional
                        An integer specifying the maximum number of iterations for the optimization algorithm.

                Returns
                -------
                ElasticNetConstrained
            The fitted ElasticNetConstrained object.


        """

        if use_coord_descent:
            if nlopt_optim is not None:
                raise ValueError(
                    "when 'use_coord_descent' is true,"
                    + "'nlopt_optim' param will not be considered"
                )
            if self.lmbd is not None:
                warnings.warn(
                    "when 'use_coord_descent' is true,"
                    + "'lmbd' param will not be considered"
                )
        else:
            if self.lmbd is None:
                raise ValueError(
                    "when 'use_coord_descent' is false, 'lmbd' param is must"
                )
            if self.alpha is None:
                raise ValueError(
                    "when 'use_coord_descent' is false, 'alpha' param is must"
                )

        self.scale_data = scale_data
        # optimization
        if not nlopt_optim:
            nlopt_optim = nlopt.LN_BOBYQA

        self.train_data = data
        n_samples = len(data)
        # data parsing using formulae
        self.dm = design_matrices(self.formula, data)

        self.y = self.dm.response.as_dataframe().values
        self.x_fixed = self.dm.common.design_matrix
        # pdb.set_trace()
        intercept_col = self.dm.common.slices["Intercept"]
        self.x_wo_intercept = (np.delete(self.x_fixed, intercept_col, axis=1),)
        self.x_wo_intercept = np.reshape(
            self.x_wo_intercept,
            (self.x_fixed.shape[0], self.x_fixed.shape[1] - 1),
        )
        # parameter lengths
        self.beta_len = self.x_fixed.shape[1]

        x0_fixed = np.zeros(self.beta_len)
        # self.param_len = self.beta_len

        # checking the box_constraints
        fixed_terms = list(self.dm.common.terms.keys())

        if nlopt_optim in [nlopt.GN_DIRECT, nlopt.GN_DIRECT_L]:
            coef_max_bound = 1e4
        else:
            coef_max_bound = np.inf

        fixed_upper_limits = np.ones(self.beta_len) * coef_max_bound
        fixed_lower_limits = np.ones(self.beta_len) * -1 * coef_max_bound

        if box_constraints:
            for term, limits in box_constraints.items():
                limits = np.array(limits)
                # TODO: raise if _x0 is inf case
                _x0 = limits[~np.isnan(limits) & ~np.isinf(limits)].mean()

                if term in fixed_terms:
                    slicer = self.dm.common.slices[term]
                    fixed_upper_limits[slicer] = limits.max()
                    fixed_lower_limits[slicer] = limits.min()
                    x0_fixed[slicer] = _x0
                else:
                    raise ValueError(
                        f""" "{term}" is not among the terms {fixed_terms}."""
                    )
                # if verbose:
                #     print(f"{term} is not there in the formula")
        # if self.alpha is None:
        #     self.scale_data = True
        if self.alpha == 0:
            self.alpha = 1e-6
        if self.scale_data:
            # pdb.set_trace()
            self.scaler = StandardScaler().fit(self.x_fixed[:, 1:])
            self.x_fixed[:, 1:] = self.scaler.transform(self.x_fixed[:, 1:])
        if use_coord_descent:
            print("ElasticNet Coord Descent")

            if box_constraints:
                print("With Constraints")
                self.beta, self.best_lmbd, self.lmbd_history = elastic_net(
                    self.x_wo_intercept,
                    np.hstack(self.y),
                    self.alpha,
                    np.array(fixed_lower_limits),
                    np.array(fixed_upper_limits),
                    tol=1e-4,
                    path_length=100,
                    epsilon=1e-4,
                )
            else:
                print("Without Constraints")
                self.beta, self.best_lmbd, self.lmbd_history = elastic_net(
                    self.x_wo_intercept,
                    np.hstack(self.y),
                    self.alpha,
                    np.ones(self.x_wo_intercept.shape[1] + 1) * -np.inf,
                    np.ones(self.x_wo_intercept.shape[1] + 1) * np.inf,
                )
        else:
            print(f"LM NLOpt: Lmbd : {self.lmbd} Alpha : {self.alpha}")
            opt = nlopt.opt(nlopt_optim, self.beta_len)
            # GN_DIRECT,GN_DIRECT_L
            # Optimization algorithms that can be used
            # NLOPT_LN_COBYLA, NLOPT_LN_BOBYQA, NLOPT_LN_NEWUOA_BOUND
            # NLOPT_LN_PRAXIS,LN_NELDERMEAD

            if (box_constraints) or (
                nlopt_optim in [nlopt.GN_DIRECT, nlopt.GN_DIRECT_L]
            ):
                opt.set_lower_bounds(list(fixed_lower_limits))
                opt.set_upper_bounds(list(fixed_upper_limits))

            opt.set_min_objective(self._objective)
            opt.set_xtol_rel(1e-8)
            # max_iter_opt = int(max(10000, 2 * np.sqrt(n_samples * self.beta_len)))
            # print("max iter: ", max_iter_opt)
            opt.set_maxeval(max_iter_opt)

            self.beta = opt.optimize(x0_fixed)
            # print("Optimization run complete")
            # print("number of function evals: ", opt.get_numevals())
            err_msg = opt.get_errmsg()
            if err_msg:
                print(err_msg)
        if self.scale_data:
            self.beta[1:] = self.scaler.transform(
                np.array(self.beta[1:]).reshape(1, -1)
            )[0]
        self.is_fitted_ = True

        if verbose:
            self._print_coeff()

        return self

    def get_coefficients(self):
        """
        Retrieves the coefficients of the fitted elastic net model.

        Returns
        -------
        dict
                A dictionary containing the elastic net coefficients of the fitted model.
        """

        check_is_fitted(self, "is_fitted_")
        fx_coef = {}
        beta = self.beta.copy()
        # if self.scale_data:
        # beta[1:] = self.scaler.transform(np.array(self.beta[1:]).reshape(1, -1))[0]
        beta = [x if abs(x) > 1e-12 else 0 for x in beta]
        for term, slc in self.dm.common.slices.items():
            fx_coef[term] = beta[slc][0]
        return fx_coef

    def _print_coeff(self):
        print("Coefficients")
        fx_coef = [["Term", "Coeff"]]
        for term, slc in self.dm.common.slices.items():
            # slc = values["cols"]
            fx_coef.append([term, self.beta[slc][0]])
        print(
            tabulate(
                fx_coef,
                headers="firstrow",
                tablefmt="psql",
                floatfmt=".4f",
            )
        )

    def _eq(self, x_fixed, beta):
        return np.dot(x_fixed, beta)

    def _objective(self, *args):

        beta = args[0]
        y_pred = self._eq(self.x_fixed, beta)
        # rho = X_j.T @ (y - y_pred + theta[j] * X_j)
        y_diff = self.y.reshape(1, -1)[0] - y_pred.reshape(1, -1)[0]
        if (self.alpha) and (self.lmbd):
            loss = (1 / (2 * len(y_pred))) * np.dot(y_diff, y_diff.T) + self.lmbd * (
                self.alpha * np.sum(np.abs(beta))
                + 0.5 * (1 - self.alpha) * np.dot(beta, beta.T)
            )
            # print("loss : ", loss, np.dot(y_diff, y_diff.T))
        else:
            raise NotImplementedError("Only regularised model is implemented")
        return loss

    def score(self, *args):
        """
        Method used to calculate the R-squared score for the model.
        It takes a variable-length argument tuple `args`, but this is ignored.

        Parameters:
        -----------
        args : tuple
                A variable-length argument tuple. This is ignored.

        Returns:
        --------
        score : float
                The R-squared score for the mixed effect model.
        """

        return r2_score(self.y, self.predict(self.train_data))

    def predict(self, data, verbose=True):
        """
        Use the fitted Elastic net model to make predictions on new data.

        Parameters:
        -----------
        data : pandas.DataFrame
                A pandas DataFrame with the same columns as the training data.
        verbose : bool
                A boolean parameter that determines whether to print warnings about unseen
                categorical values in the new data. Default is True.

        Returns:
        --------
        y_pred_random + y_pred_fixed : numpy.ndarray
                A numpy array of predicted response variable values for the new data.
        """

        pd.options.mode.chained_assignment = None
        check_is_fitted(self, "is_fitted_")

        x_fixed = self.dm.common.evaluate_new_data(data).design_matrix
        # if self.scale_data:
        #     x_fixed[:, 1:] = self.scaler.transform(x_fixed[:, 1:])
        y_pred_fixed = self._eq(x_fixed, self.beta)
        return y_pred_fixed


class LinearRegressionConstrained(BaseEstimator):
    """
    Linear regression model with box constraints on the coefficients.

    Parameters
    ----------
    formula : str
        A string representing the formula for the linear regression model.

    Attributes
    ----------
    beta : array-like, shape (n_features,)
        The coefficients of the fitted linear regression model.
    train_data : pandas.DataFrame
        The training data used to fit the model.
    scale_data : bool
        Whether or not to scale the data before fitting the model.
    dm : patsy.design_info.DesignMatrix
        The design matrix used in the linear regression model.
    """

    def __init__(self, formula):
        self.formula = formula

    def fit(
        self,
        data,
        box_constraints=None,
        scale_data=False,
        nlopt_optim=None,
        verbose=False,
        max_iter_opt=5000,
        x0=[],
    ):
        """
        Fit the linear regression model to the training data.

        Parameters
        ----------
        data : pandas.DataFrame
            The training data to fit the model to.
        box_constraints : dict, optional
            A dictionary containing the box constraints on the coefficients.
        scale_data : bool, optional
            Whether or not to scale the data before fitting the model.
        nlopt_optim : callable, optional
            The optimization algorithm to use. Default is nlopt.LN_BOBYQA.
        verbose : bool, optional
            Whether or not to print the coefficients after fitting the model.
        max_iter_opt : int, optional
            The maximum number of iterations to run the optimization algorithm.
        x0 : array-like, shape (n_features,), optional
            The initial guess for the coefficients.

        Returns
        -------
        self : LinearRegressionConstrained
            The fitted linear regression model.

        Raises
        ------
        ValueError
            If x0 is not the same length as the number of features.
        """

        self.scale_data = scale_data
        # optimization
        if not nlopt_optim:
            nlopt_optim = nlopt.LN_BOBYQA

        self.train_data = data
        n_samples = len(data)
        # data parsing using formulae
        self.dm = design_matrices(self.formula, data)

        self.y = self.dm.response.as_dataframe().values
        self.x_fixed = self.dm.common.design_matrix
        # pdb.set_trace()
        intercept_col = self.dm.common.slices["Intercept"]
        self.x_wo_intercept = (np.delete(self.x_fixed, intercept_col, axis=1),)
        self.x_wo_intercept = np.reshape(
            self.x_wo_intercept,
            (self.x_fixed.shape[0], self.x_fixed.shape[1] - 1),
        )
        # parameter lengths
        self.beta_len = self.x_fixed.shape[1]
        if not x0:
            x0_fixed = np.zeros(self.beta_len)
        else:
            if len(x0) != self.beta_len:
                raise ValueError(
                    f"x0 Length mismatch: expecting {self.beta_len} got {len(x0)}"
                )
            else:
                x0_fixed = x0

        # self.param_len = self.beta_len

        # checking the box_constraints
        fixed_terms = list(self.dm.common.terms.keys())

        if nlopt_optim in [nlopt.GN_DIRECT, nlopt.GN_DIRECT_L]:
            coef_max_bound = 1e4
        else:
            coef_max_bound = np.inf

        fixed_upper_limits = np.ones(self.beta_len) * coef_max_bound
        fixed_lower_limits = np.ones(self.beta_len) * -1 * coef_max_bound

        if box_constraints:
            for term, limits in box_constraints.items():
                limits = np.array(limits)
                # TODO: raise if _x0 is inf case
                _x0 = limits[~np.isnan(limits) & ~np.isinf(limits)].mean()

                if term in fixed_terms:
                    slicer = self.dm.common.slices[term]
                    fixed_upper_limits[slicer] = limits.max()
                    fixed_lower_limits[slicer] = limits.min()
                    x0_fixed[slicer] = _x0
                else:
                    raise ValueError(
                        f""" "{term}" is not among the terms {fixed_terms}."""
                    )
                # if verbose:
                #     print(f"{term} is not there in the formula")

        if self.scale_data:
            # pdb.set_trace()
            self.scaler = StandardScaler().fit(self.x_fixed[:, 1:])
            self.x_fixed[:, 1:] = self.scaler.transform(self.x_fixed[:, 1:])

        opt = nlopt.opt(nlopt_optim, self.beta_len)
        # GN_DIRECT,GN_DIRECT_L
        # Optimization algorithms that can be used
        # NLOPT_LN_COBYLA, NLOPT_LN_BOBYQA, NLOPT_LN_NEWUOA_BOUND,
        # NLOPT_LN_PRAXIS,LN_NELDERMEAD

        if (box_constraints) or (nlopt_optim in [nlopt.GN_DIRECT, nlopt.GN_DIRECT_L]):
            opt.set_lower_bounds(list(fixed_lower_limits))
            opt.set_upper_bounds(list(fixed_upper_limits))

        opt.set_min_objective(self._objective)
        opt.set_xtol_rel(1e-8)
        # if not max_iter_opt:
        # max_iter_opt = 5000 #int(max(5000, 2 * np.sqrt(n_samples * self.beta_len)))
        # print("max iter: ", max_iter_opt)
        opt.set_maxeval(max_iter_opt)

        self.beta = opt.optimize(x0_fixed)
        # print("Optimization run complete")
        # print("number of function evals: ", opt.get_numevals())
        err_msg = opt.get_errmsg()
        if err_msg:
            print(err_msg)

        self.is_fitted_ = True

        if verbose:
            self._print_coeff()

        return self

    def get_coefficients(self):
        """
        Retrieves the coefficients of the fitted linear regression model.

        Returns
        -------
        dict
                A dictionary containing the linear regression coefficients of the fitted model.
        """

        check_is_fitted(self, "is_fitted_")
        fx_coef = {}
        beta = self.beta.copy()
        if self.scale_data:
            beta[1:] = self.scaler.transform(np.array(self.beta[1:]).reshape(1, -1))[0]
        beta = [x if abs(x) > 1e-12 else 0 for x in beta]
        for term, slc in self.dm.common.slices.items():
            fx_coef[term] = beta[slc][0]
        return fx_coef

    def _print_coeff(self):
        print("Coefficients")
        fx_coef = [["Term", "Coeff"]]
        for term, slc in self.dm.common.slices.items():
            # slc = values["cols"]
            fx_coef.append([term, self.beta[slc][0]])
        print(
            tabulate(
                fx_coef,
                headers="firstrow",
                tablefmt="psql",
                floatfmt=".4f",
            )
        )

    def _eq(self, x_fixed, beta):
        return np.dot(x_fixed, beta)

    def _objective(self, *args):

        beta = args[0]
        y_pred = self._eq(self.x_fixed, beta)
        # rho = X_j.T @ (y - y_pred + theta[j] * X_j)
        y_diff = self.y.reshape(1, -1)[0] - y_pred.reshape(1, -1)[0]

        loss = (1 / (2 * len(y_pred))) * np.dot(y_diff, y_diff.T)

        return loss

    def score(self, *args):
        """
        Method used to calculate the R-squared score for the mixed effect model.
        It takes a variable-length argument tuple `args`, but this is ignored.

        Parameters:
        -----------
        args : tuple
                A variable-length argument tuple. This is ignored.

        Returns:
        --------
        score : float
                The R-squared score for the mixed effect model.
        """

        return r2_score(self.y, self.predict(self.train_data))

    def predict(self, data, verbose=True):
        """
        Use the fitted Elastic net model to make predictions on new data.

        Parameters:
        -----------
        data : pandas.DataFrame
                A pandas DataFrame with the same columns as the training data.
        verbose : bool
                A boolean parameter that determines whether to print warnings about unseen
                categorical values in the new data. Default is True.

        Returns:
        --------
        y_pred_fixed : numpy.ndarray
                A numpy array of predicted response variable values for the new data.
        """

        pd.options.mode.chained_assignment = None
        check_is_fitted(self, "is_fitted_")

        x_fixed = self.dm.common.evaluate_new_data(data).design_matrix
        if self.scale_data:
            x_fixed[:, 1:] = self.scaler.transform(x_fixed[:, 1:])
        y_pred_fixed = self._eq(x_fixed, self.beta)
        return y_pred_fixed


@jit(nopython=True)
def elastic_net(
    X,
    y,
    alpha,
    lower_limits=[-np.inf],
    upper_limits=[np.inf],
    tol=1e-3,
    path_length=100,
    epsilon=1e-4,
):
    """The Elastic Net Regression model with intercept term.
    Intercept term included via design matrix augmentation.
    Pathwise coordinate descent with co-variance updates is applied.
    Path from max value of the L1 tuning parameter to input tuning parameter value.
    Features must be standardized (centered and scaled to unit variance)
    Params:
        X - NumPy matrix, size (N, p), of standardized numerical predictors
        y - NumPy array, length N, of numerical response
        l1 - L1 penalty tuning parameter (positive scalar)
        l2 - L2 penalty tuning parameter (positive scalar)
        tol - Coordinate Descent convergence tolerance
        (exited if change < tol)
        path_length - Number of tuning parameter values to include in path
        (positive integer)
    Returns:
        NumPy array, length p, of fitted model coefficients
    Reference: https://github.com/wyattowalsh/regularized-linear-regression-deep-dive/
    blob/master/src/linear_regression.py
    """

    m, n = np.shape(X)
    X_intercept = np.hstack((np.ones((len(X), 1)), X))
    B_star = np.zeros((n + 1))
    s = np.empty_like(B_star)
    if not alpha:
        alpha = 1e-6
    l_max = max(list(np.abs(np.dot(np.transpose(X), y)))) / m / max(alpha, 1e-3)
    # l_path = np.geomspace(l_max, min(l_max * epsilon, 0.01), path_length)
    l_path = np.power(
        10,
        np.linspace(np.log10(l_max), np.log10(min(l_max * epsilon, 0.01)), path_length),
    )
    # print(f"Lmbd:{l_max} {l_max * epsilon}")
    LOWER = 1e-2
    UPPER = 1e9
    lmbd_history = []
    diverge = False
    reached_max_iteration = False
    for i in range(path_length):
        iteration = 0
        while True:  # 30
            B_s = B_star.copy()
            for j in range(n, -1, -1):
                if j == 0:
                    B_star[0] = np.mean(y) - np.dot(np.sum(X, axis=0) / len(X), B_s[1:])
                else:
                    k = np.where(B_s != 0)[0]
                    update = (1 / m) * (
                        (
                            np.dot(X_intercept[:, j], y)
                            - np.dot(
                                np.dot(X_intercept[:, j], X_intercept[:, k]),
                                B_s[k],
                            )
                        )
                    ) + B_s[j]

                    if np.abs(update) > UPPER:
                        update = np.sign(update) * UPPER
                    elif np.abs(update) < LOWER:
                        update = np.sign(update) * LOWER  # 50

                    B_star[j] = (
                        np.sign(update) * max(np.abs(update) - l_path[i] * alpha, 0)
                    ) / (1 + (l_path[i] * (1 - alpha)))
                if B_star[j] > upper_limits[j]:  # 57
                    B_star[j] = upper_limits[j]
                elif B_star[j] < lower_limits[j]:
                    B_star[j] = lower_limits[j]

            if np.all(np.abs(B_s - B_star) < tol):

                lmbd_history.append((l_path[i], list(np.round(np.copy(B_star), 2, s))))
                break
            else:
                iteration += 1
                if iteration > 50000:
                    print("*", l_path[i])
                    lmbd_history.append(
                        (l_path[i], list(np.round(np.copy(B_star), 2, s)))
                    )
                    reached_max_iteration = True
                    break
                if np.isnan(max(B_star)):
                    diverge = True
                    print("**")
                    break
        if diverge:
            print("Solution did not converge")
            break
        if reached_max_iteration:
            print("Reached maximum iterations, stopping exploration")
            break
    delta = []
    for i in range(len(lmbd_history) - 1):
        beta0, beta1 = lmbd_history[i][1], lmbd_history[i + 1][1]
        delta_loop = []
        for i in range(X.shape[1]):

            delta_loop.append((X[:, i] @ X[:, i]) * (beta0[i] - beta1[i]) ** 2)
        delta.append(max(delta_loop))
    best_lmbd_index = sum([x > 10**-4 * max(delta) for x in delta])
    best_lmbd, coefs = lmbd_history[best_lmbd_index]
    return coefs, best_lmbd, lmbd_history
