import numpy as np
import pandas as pd
from hvplot import hvPlot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold, cross_val_score


class FeatureElimination:
    """Returns a curve that shows the accuracy of the model as more features are added into it.

    Recursive feature elimination (RFE) is a feature selection method that fits a model and removes the weakest
    feature (or features) until the specified number of features is reached. Features are ranked by the model’s coef_
    or feature_importances_ attributes, and by recursively eliminating a small number of features per loop,
    RFE attempts to eliminate dependencies and co-linearity that may exist in the model.
    """

    def __init__(self, is_classification, model=None, cv=None, score=None, **kwargs):
        """Class initializer.

        Parameters
        ----------
        is_classification : `Boolean`
            whether the model you're using is classification or regression. or if the target
            is `categorical` or `numerical`.
        model : a scikit-learn estimator
            An object that implements `fit` and provides information about the
            relative importance of features with either a `coef` or
            `feature_importances_` attribute.
        cv : `int`
            cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
                - None, to use the default 3-fold cross-validation,
                - integer, to specify the number of folds.
                - An object to be used as a cross-validation generator.
                - An iterable yielding train/test splits.
        score : `string`
            callable or None, optional, default: None
        kwargs : `dict`
            Keyword arguments that are passed to the base class
        """
        if is_classification:
            self.model = model or RandomForestClassifier(n_estimators=100)
            self.cv = cv or StratifiedKFold(n_splits=5)
            self.score = score or "roc_auc"
        else:
            self.model = model or LassoCV()
            self.cv = cv or 5
            self.score = score or "neg_mean_absolute_error"
        self.step = 0.1

    def fit(self, X, y):
        """Fits x and y vars.

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
        n_features = X.shape[1]

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)

        rfe = RFE(self.model, step=step)
        self.n_feature_subsets_ = np.arange(1, n_features + step, step)

        cv_params = {"cv": self.cv, "scoring": self.score}
        scores = []
        for n_features_to_select in self.n_feature_subsets_:
            rfe.set_params(n_features_to_select=n_features_to_select)
            scores.append(cross_val_score(rfe, X, y, **cv_params))
        self.cv_scores_ = np.array(scores)
        return self

    def get_plot_data(self):
        """Gets plot data.

        Returns
        -------
        plot : `pandas dataframe`
        """

        x = self.n_feature_subsets_
        means = self.cv_scores_.mean(axis=1)
        sigmas = self.cv_scores_.std(axis=1)
        # plt.fill_between(x, means - sigmas, means + sigmas, alpha=0.25)
        data = pd.DataFrame(
            {
                "no of features": x,
                "avg cv score": means,
                "sigmas": sigmas,
                "means-sigmas": means - sigmas,
                "means+sigmas": means + sigmas,
            }
        )

        return data

    def get_plot(self):
        """Gets the plot.

        Returns
        -------
        plot : `hvplot`
            It plots an ideal `RFECV` curve, the curve jumps to an excellent accuracy when the informative
            features are captured, then gradually decreases in accuracy as the non informative features are added into
            the model. The shaded area represents the variability of cross-validation, one standard deviation above and
            below the mean accuracy score drawn by the curve.
        """
        data = self.get_plot_data()

        plot = (
            hvPlot(data).area("no of features", "means-sigmas", "means+sigmas")
            * hvPlot(data).line(x="no of features", y="avg cv score", c="pink")
            * hvPlot(data).scatter(x="no of features", y="avg cv score", c="k")
        )

        return plot


class PermutedFeatureImportance:
    """Permuted Feature Importance class."""

    pass
