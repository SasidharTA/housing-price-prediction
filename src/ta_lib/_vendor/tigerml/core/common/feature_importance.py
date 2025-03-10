"""Feature Importance."""
import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from hvplot import hvPlot
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from tigerml.core.utils._lib import fail_gracefully
from tigerml.core.utils.modeling import Algo

_LOGGER = logging.getLogger(__name__)

algo_object = Algo()


class ModelFeatureImportance:
    """Returns a bar plot of feature importance given by the model.

    Displays the most informative features in a model by showing a bar chart
    of features ranked by their importances. Although primarily a feature
    engineering mechanism, this visualizer requires a model that has either a
    `coef_` or `feature_importances_` parameter after fit.
    """

    def __init__(
        self,
        model=None,
        cv=None,
        score=None,
        is_classification=None,
        algo=None,
        **kwargs,
    ):
        """Returns a bar plot of feature importance given by the model.

        Parameters
        ----------
        model : Estimator
            A Scikit-Learn estimator that learns feature importances.
        cv : `int`
            cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
        score : `string`
            Scoring function
        is_classification : `Boolean`
            True if dependent variable is categorical, Classification problem,
            else False.
        kwargs : `dict`
            Keyword arguments that are passed to the base class.
        """
        # if "ax" in kwargs:
        #     self.ax = kwargs["ax"]
        # else:
        #     self.fig, self.ax = plt.subplots()
        self.estimator = model
        self.is_classification = is_classification
        self.cv = cv
        self.score_func = score
        self.error = None
        # Algo Reference
        self.algo = algo
        if (algo is None and hasattr(model, "predict_proba")) or self.is_classification:
            self.algo = algo_object.classification

    def _create_model(
        self, is_classification=None, model=None, cv=None, score=None, labels=None
    ):
        """Creating a model.

        is_classification : `Boolean`
            True if dependent variable is categorical, Classification problem,
            else False.
        model : Estimator
            A `Scikit-Learn` estimator that learns feature importances.
        cv : `int`
            cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
        score : `string`
            Scoring function
        labels : `list`, default: None
            A list of feature names to use. If a DataFrame is passed to fit and
            features is None, feature names are selected as the column names.
        """
        if model is None:
            if algo_object.is_classification(self.algo):
                cv = cv or StratifiedKFold(n_splits=5)
                score = score or "accuracy"
                model = LogisticRegressionCV(
                    solver="lbfgs", cv=cv, scoring=score, max_iter=15000
                )
            elif algo_object.is_classification(self.algo) is False:
                cv = cv or 5
                model = LassoCV(cv=cv, max_iter=30000)
                score = score or "neg_mean_absolute_error"
            else:
                raise Exception(
                    "Should pass either model or is_classification as input."
                )
        if "sklstatsmodellogit" in str(model).lower():
            cv = cv or StratifiedKFold(n_splits=5)
            score = score or "accuracy"
            model = LogisticRegressionCV(
                solver="lbfgs", cv=cv, scoring=score, max_iter=15000
            )
        self.model = model
        self.estimator = model

    def _get_feature_importance(self, labels):
        # Get the feature importances from the model
        feature_importances_ = None
        for attr in ("feature_importances_", "coef_"):
            try:
                feature_importances_ = getattr(self.estimator, attr)
            except AttributeError:
                continue
        if feature_importances_ is None:
            raise Exception(
                "could not find feature importances param for {}".format(
                    self.estimator.__class__.__name__
                )
            )

        if feature_importances_.ndim > 1:
            feature_importances_ = np.mean(feature_importances_, axis=0)

        # Normalize features relative to the maximum
        maxv = np.abs(feature_importances_).max()
        feature_importances_ /= maxv
        feature_importances_ *= 100.0

        # Create labels for the feature importances
        features_ = np.array(labels)

        return feature_importances_, features_

    def fit(self, X, y):
        """Using fit taking X and y as arguments.

        Parameters
        ----------
        X : pd.DataFrame, independent variables.
        y : pd.Series, dependent variable.
        """
        # initialize the model
        if self.is_classification is not None or self.estimator:
            self._create_model(
                self.is_classification,
                model=self.estimator,
                labels=X.columns,
                cv=self.cv,
                score=self.score_func,
            )
        else:
            from tigerml.core.utils.pandas import is_discrete

            is_classification = is_discrete(y)
            self._create_model(is_classification, labels=X.columns)
        # fit the model
        self.model.fit(X, y)
        # get feature importance
        feature_importances_, features_ = self._get_feature_importance(labels=X.columns)
        data = (
            td.DataFrame(feature_importances_)
            .set_index(features_)
            .merge(
                td.DataFrame(X.abs().mean().rename("mean")),
                left_index=True,
                right_index=True,
            )
        )
        feature_importances_ = data.iloc[:, 0].mul(data.iloc[:, 1]).to_numpy()
        sort_idx = np.argsort(feature_importances_)
        self.features_ = features_[sort_idx]
        self.feature_importances_ = feature_importances_[sort_idx]
        return self

    def score(self, X, y):
        """Returns model score R-Squared value if it is a `regression model` else accuracy score.

        Parameters
        ----------
        X : pd.DataFrame, independent variables.
        y : pd.Series, dependent variable.

        Returns
        -------
        score : `float`
            model score R-Squared value if it is a regression model
            else accuracy score
        """
        return self.model.score(X, y)

    def get_plot_data(self, n=20):
        """Returns Feature importances dataframe."""
        feat_imp = pd.DataFrame(
            self.feature_importances_, index=self.features_, columns=["importance"]
        )
        feat_imp["abs"] = abs(feat_imp["importance"])
        feat_imp = feat_imp.nlargest(min([len(self.features_), n]), "abs")
        feat_imp.drop(["abs"], inplace=True, axis=1)
        feat_imp.sort_values("importance", ascending=True, inplace=True)
        return feat_imp

    @fail_gracefully()
    def get_plot(self, n=20):
        """Draws the `feature importances` as a bar chart; called from `fit`."""
        if self.error is None:
            try:
                if (
                    "sklstatsmodellogit" in str(self.estimator).lower()
                    or "sklstatsmodelols" in str(self.estimator).lower()
                    or hasattr(self.estimator, "coef_")
                ):
                    ylabel = "Feature Importance [determined by coeff * mean(x)]"
                else:
                    ylabel = "Feature Importance"
                plot = hvPlot(self.get_plot_data(n=n)).bar(
                    invert=True,
                    rot=0,
                    legend="top",
                    stacked=False,
                    ylabel=ylabel,
                    xlabel="Features",
                    title="Feature Importances from {}".format(
                        self.estimator.__class__.__name__
                    ),
                )
                return plot
            except AttributeError as e:  # no model_importances_
                return str(e)
        else:
            return str(self.error)
