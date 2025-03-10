import logging
import matplotlib.pyplot as plt
import pandas as pd
from hvplot import hvPlot
from scipy.stats import pearsonr
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    mutual_info_classif,
    mutual_info_regression,
)

_LOGGER = logging.getLogger(__name__)


class FeatureCorrelation:
    """Returns a plot of pearson correlation of feature columns with the target."""

    def __init__(
        self, is_classification
    ):  # add is_classification. Can be an issue if the target is categorical and correlation won't
        # give us right interpretation.
        pass

    def fit(self, X, y):  # Have added X and y in __init__ function
        """Returns updated object.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe of independent variables
        y : pd.Series
            A series containing the dependent variable.

        Returns
        -------
        self : After initiating X and y inputs
        """
        self.X = X
        self.y = y
        # self.model.fit(self.X, self.y, **kwargs)
        return self

    def get_plot_data(self):
        """Returns a pandas dataframe with pearson correlation of independent variables with dependent variable."""
        feature_target_corr = {}
        for col in self.X.columns:
            feature_target_corr[col] = pearsonr(self.X[col], self.y)[0]
        corr = pd.DataFrame.from_dict(feature_target_corr, orient="index")
        corr = corr.rename(
            columns=dict(zip(list(corr.columns), ["Pearson_correlation_with_Target"]))
        )
        if corr["Pearson_correlation_with_Target"].isnull().sum() > 0:
            removed_features = corr.index[
                corr["Pearson_correlation_with_Target"].isnull()
            ].tolist()
            _LOGGER.warning(
                "User Warning: Removed {} features {} whose pearson's correlation coefficient is nan".format(
                    len(removed_features), removed_features
                )
            )
            corr = corr.dropna(axis=0)
        corr.sort_values(by="Pearson_correlation_with_Target", inplace=True)
        return corr

    def get_plot(self):
        """Returns an interactive bar plot of pearson correlation of independent variables with dependent variable.

        Returns
        -------
        plot : returns feature's correlation with the dependent variable called from fit.
        """
        corr = self.get_plot_data()
        plot = hvPlot(corr).bar(
            invert=True, title="Feature Correlation with Target Function"
        )
        return plot


class MutualInformation:
    """Returns an information gain plot to dependent variable."""

    def __init__(self, is_classification):
        """Class initializer.

        Parameters
        ----------
        is_classification: `bool`
            whether the model you're using is classification or regression. or if the target
            is `categorical` or `numerical`.
        """
        self.is_classification = is_classification
        if is_classification:
            self.method = mutual_info_classif
        else:
            self.method = mutual_info_regression

    def fit(self, X, y, **kwargs):
        """Fits x and y variables.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe containing independent variables
        y : pd.Series
            A series containing the dependent variable
        kwargs : `dict`
            Keyword arguments passed to the fit method of the estimator.

        Returns
        -------
        self : `visualizer`
            The fit method returns self to support pipelines.
        """
        self.X = X
        self.scores_ = self.method(X, y, **kwargs)
        return self

    def get_plot_data(self):
        """Returns a pandas dataframe with information gain to dependent variable from all IDVS."""
        data = pd.DataFrame(self.scores_, index=self.X.columns)
        data.columns = ["Mutual_information_with_Target"]
        data.sort_values(by="Mutual_information_with_Target", inplace=True)
        return data

    def get_plot(self):
        """Draws the information gain to dependent variable from all IDVs, called from fit."""
        # FIX: pivot -> plot
        data = self.get_plot_data()
        # data = self.get_pivot_data()

        plot = hvPlot(data).bar(invert=True, title="Mutual Information with Target")
        return plot


class FScore(GenericUnivariateSelect):
    """Returns a bar plot of FScores.

    The F score is used to measure a test’s accuracy, and it balances the use of precision and recall to do it.
    """

    def __init__(self, is_classification):
        """Class initializer.

        Parameters
        ----------
        is_classification: `bool`
            whether the model you're using is classification or regression. or if the target
            is `categorical` or `numerical`.
        """
        self.is_classification = is_classification
        if is_classification:
            from sklearn.feature_selection import f_classif

            score_func = f_classif
        else:
            from sklearn.feature_selection import f_regression

            score_func = f_regression
        super().__init__(score_func=score_func)

    def fit(self, X, y):
        """Returns fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe containing independent variables
        y : pd.Series
            A series containing the dependent variable

        Returns
        -------
        estimator : fitted model which we later use to create scores to use in get_plot.
        """
        self.X = X
        return super().fit(X, y)

    def get_plot(self):
        """Returns a bar plot with FScore for every feature."""
        scores = self.scores_
        scores = [score / max(scores) for score in scores]
        plot = hvPlot(pd.DataFrame({"scores": scores, "feat": self.X.columns})).bar(
            x="feat", y="scores", invert=True
        )
        return plot


class Chi2(GenericUnivariateSelect):
    """Returns a bar plot of Chi-square test scores for IDVs.

    Chi-squared test is used to determine whether there is a statistically significant difference (i.e., a magnitude
    of difference that is unlikely to be due to chance alone) between the expected frequencies and the observed
    frequencies in one or more categories of a so-called contingency table.
    """

    def __init__(self, is_classification):
        """Class initializer.

        Parameters
        ----------
        is_classification: `bool`
            whether the model you're using is classification or regression. or if the target
            is `categorical` or `numerical`.
        """
        self.is_classification = is_classification
        if is_classification:
            from sklearn.feature_selection import chi2

            score_func = chi2
            super().__init__(score_func=score_func)
        else:
            self.error = "Cannot generate for continuous dependant variable"

    def fit(self, X, y):
        """Fits x and y values.

        Parameters
        ----------
        X : pd.DataFrame
            A dataframe containing independent variables
        y : pd.Series
            A series containing the dependent variable

        Returns
        -------
        self : visualizer
            The fit method returns self to support pipelines.
        """
        if self.is_classification:
            try:
                self.X = X
                super().fit(X, y)
            except Exception as e:
                self.error = "Cannot generate. Error - {}".format(e)
        return self

    def get_plot(self):
        """Returns a bar plot with Chi-squared test scores for every feature if model has attribute "scores" else error."""
        if hasattr(self, "scores_"):
            scores = self.scores_
            scores = [score / max(scores) for score in scores]
            plot = hvPlot(
                pd.DataFrame({"scores": scores, "feat": self.X.columns}).bar(
                    x="feat", y="scores", invert=True
                )
            )
            return plot
        else:
            return self.error
