import gc
import logging
import numpy as np
import pandas as pd
from holoviews.operation.datashader import dynspread
from tigerml.core.plots import hvPlot
from tigerml.core.preprocessing.imputer import Imputer
from tigerml.core.reports import create_report
from tigerml.core.utils import (
    append_file_to_path,
    fail_gracefully,
    measure_time,
    time_now_readable,
)

_LOGGER = logging.getLogger(__name__)


class KeyDriversMixin:
    """Calls all the classes defined in key_drivers to be used in report.

    Mixins are a sort of class that is used to "mix in" extra properties and methods into a class. This allows you
    to create classes in a compositional style.
    """

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def get_feature_scores(
        self,
        y=None,
        features=None,
        scores=["feature_correlation"],
        quick=False,
        data=None,
    ):
        """Returns the dictionary which contains correlation and information gain from the dependent variable.

        Parameters
        ----------
        y : pd.Series
            default: None, will take it from the classes it's being called from.
        scores : `list`
            contains string objects in the list where you can define the plots you need.
        quick : `bool`
            default: `False`, if `True` "Mutual Information" won't be plotted.

        Returns
        -------
        plot_dict : `dict`
            a dictionary of plots from key_drivers.
        """
        if y is not None and y != self._current_y:
            self._set_current_y(y)
        df = data or self.data
        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols
        X = df[self.get_numeric_columns(df[features])]
        y = df[self._current_y]
        from . import Chi2, FeatureCorrelation, FScore, MutualInformation

        plotters = {
            "feature_correlation": FeatureCorrelation,
            "mutual_information": MutualInformation,
            # 'f_score': FScore,
            # 'chi2_score': Chi2
        }
        plots_dict = {}
        if self.is_classification:
            scores += ["mutual_information"]
        if scores:
            plotters = {key: plotters[key] for key in scores}
        if quick and "mutual_information" in plotters:
            plotters.pop("mutual_information")
        for plot_name in plotters:
            try:
                obj = plotters[plot_name](self.is_classification).fit(X, y)
                plot = obj.get_plot()
                # from tigerml.core.plots import autosize_plot
                # plot = autosize_plot(plot)
                del obj
                gc.collect()
            except Exception as e:
                plot = "Cannot generate. Error - {}".format(e)
            plots_dict[plot_name] = plot
        del X, y
        return plots_dict

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def feature_elimination_analysis(self, y=None, features=None):
        """Returns the feature elimination plot which we create using Recursive feature elimination.

        Parameters
        ----------
        y : pd.Series, default : `None`
            By default definition in class are used.
        features : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.

        Returns
        -------
        plot :
            An ideal RFECV curve. The curve jumps to an excellent accuracy when the informative
            features are captured, then gradually decreases in accuracy as the non informative features are added into
            the model. The shaded area represents the variability of cross-validation, one standard deviation above and
            below the mean accuracy score drawn by the curve.
        """
        if y and y != self._current_y:
            assert isinstance(y, str)
            self._set_current_y(y)
        from . import FeatureElimination

        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols
        X = self.data[self.get_numeric_columns(self.data[features])]
        y = self.data[self._current_y]
        plot = FeatureElimination(self.is_classification).fit(X, y).get_plot()
        del X, y
        gc.collect()
        return plot

    # @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def feature_importances(self, y=None, features=None, quick=False):
        """Return model `feature importances` for IDVs.

        Following Scikit-learn models are used as baseline models to calculate feature importances:
        1. Classification:  LogisticRegressionCV
        2. Regression: LassoCV

        Parameters
        ----------
        y : pd.Series, default : `None`
            By default definition in class are used.
        features : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        quick : `bool`
            default : `False`, doesn't plot heavy plots which take longer time to compute.

        Returns
        -------
        plot : feature importance plot from both model and shap.
        """
        if y is not None and y != self._current_y:
            assert isinstance(y, str)
            self._set_current_y(y)
        from tigerml.core.common import ModelFeatureImportance

        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols

        imputer = Imputer()
        X = self.data[self.get_numeric_columns(self.data[features])]
        X = imputer.fit_transform(X)
        X = pd.DataFrame(X, columns=imputer.get_feature_names())
        X = X.drop(imputer.drop_cols_, axis=1)

        y = self.data[self._current_y]
        feature_importances = {}
        if self.is_classification:
            algo = "classification"
        else:
            algo = "regression"
        import tigerml.core.dataframe as td

        vizer = ModelFeatureImportance(is_classification=self.is_classification)
        vizer._create_model()
        vizer.estimator.fit(X, y)
        feature_importances_, features_ = vizer._get_feature_importance(
            labels=X.columns
        )
        data = (
            td.DataFrame(feature_importances_)
            .set_index(features_)
            .merge(
                td.DataFrame(X.abs().mean().rename("mean")),
                left_index=True,
                right_index=True,
            )
        )
        if hasattr(vizer.estimator, "coef_"):
            feature_importances_ = data.iloc[:, 0].mul(data.iloc[:, 1]).to_numpy()
        elif hasattr(vizer.estimator, "feature_importances_"):
            feature_importances_ = data.iloc[:, 0].to_numpy()
        sort_idx = np.argsort(feature_importances_)
        vizer.features_ = features_[sort_idx]
        vizer.feature_importances_ = feature_importances_[sort_idx]
        feature_importances["from_model"] = vizer.get_plot()
        if not quick:
            if self.is_classification:
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(solver="lbfgs")
            else:
                from sklearn.linear_model import LassoCV

                model = LassoCV()
            try:
                from tigerml.model_eval.plotters.interpretation import (
                    ModelInterpretation,
                    sample_data,
                )
            except ImportError:
                # If only tigerml.eda is installed skip SHAP plot computation
                _LOGGER.error(
                    "To include SHAP value plot, make sure tigerml.model_eval is installed."
                )
                return feature_importances
            model.fit(X, y)
            interpreter = ModelInterpretation(model=model, algo=algo, x_train=X)
            interpreter._shap_fit()
            x_sampled = sample_data(X, 1000)
            shap_values = interpreter._shap_score(x_sampled, get_expected_value=False)
            feature_importances["shap_values"] = interpreter.shap_distribution(
                shap_values=shap_values, X=x_sampled
            )
        del X, y
        gc.collect()
        return feature_importances

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def get_pca_analysis(self, y=None, features=None):
        """Return scatter plots of y vs first two PCA components.

        Parameters
        ----------
        y : pd.Series, default : `None`
            By default definition in class are used.
        features : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        """
        # FIXME: why only two components, make this configurable. Limit # variables, sequence plots correctly.
        if y and y != self._current_y:
            self._set_current_y(y)
        pca_result = {}
        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols

        imputer = Imputer()
        X = self.data[list(set(self.get_numeric_columns()) & set(features))]
        X = imputer.fit_transform(X)
        X = pd.DataFrame(X, columns=imputer.get_feature_names())
        X = X.drop(imputer.drop_cols_, axis=1)
        if self.is_classification:
            y = self.data[self._current_y].astype(int)
        else:
            y = self.data[self._current_y]
        from . import PCAProjection

        pca_plotter = PCAProjection()
        # visualizer = PCADecomposition()
        viz = pd.DataFrame(pca_plotter.fit_transform(X, y))
        pca_proj = pd.DataFrame({"PC1": viz[0], "PC2": viz[1], "color_coding": y})
        pca_proj["color_coding"] = pca_proj["color_coding"].astype(y.dtype)
        pca_result["pca_projection"] = dynspread(
            hvPlot(pca_proj).scatter(x="PC1", y="PC2", c="color_coding", datashade=True)
        )
        pca_result["correlation_with_dimension_2 (Y)"] = pca_plotter.plots[1]
        pca_result["correlation_with_dimension_1 (X)"] = pca_plotter.plots[0]
        del X, y
        return pca_result

    # @fail_gracefully(_LOGGER)
    # @measure_time(_LOGGER)
    # def _get_tsne_projection(self, y=None, features=None):
    #     """
    #
    #     Parameters
    #     ----------
    #     y : pd.Series, default : `None`
    #         By default definition in class are used.
    #     features : list, default=[]
    #         list of columns in the dataframe for analysis. By default all are used.
    #     """
    #     if y and y != self._current_y:
    #         self._set_current_y(y)
    #     manifold_result = {}
    #     if features:
    #         features = list(set(features) & set(self.x_cols))
    #     else:
    #         features = self.x_cols
    #     X = self.data[list(set(self.get_numeric_columns()) & set(features))]
    #     y = self.data[self._current_y]
    #     from . import TSNEProjection
    #
    #     tsne_plotter = TSNEProjection().fit_transform(X, y)
    #     # visualizer = PCADecomposition()
    #     viz = pd.DataFrame(tsne_plotter.fit_transform(X, y))
    #     tsne_proj = pd.DataFrame({"PC1": viz[0], "PC2": viz[1], "color_coding": y})
    #     tsne_proj["color_coding"] = tsne_proj["color_coding"].astype(y.dtype)
    #     manifold_result["tsne_projection"] = dynspread(
    #         hvPlot(tsne_proj).scatter(
    #             x="PC1", y="PC2", c="color_coding", datashade=True
    #         )
    #     )
    #     manifold_result["correlation_with_dimension_2 (Y)"] = tsne_plotter.plots[1]
    #     manifold_result["correlation_with_dimension_1 (X)"] = tsne_plotter.plots[0]
    #     del X, y
    #     return manifold_result

    def key_drivers(
        self, y=None, features=None, quick=True, save_as=None, save_path=""
    ):
        """Univariate analysis for the columns.

        Generate summary_stats, distributions, feature importances and normality tests for columns.

        Following Scikit-learn models are used as baseline models to calculate feature importances:
        1. Classification:  LogisticRegressionCV
        2. Regression: LassoCV

        Parameters
        ----------
        y : pd.Series, default=None
            By default definition in class are used.
        features : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        save_as : str, default=None
            You need to pass only extension here ".html" or ".xlsx". If none results will not be saved.
        save_path : str, default=''
            Location where report to be saved. By default report saved in working directory.
            This should be without extension just complete path where you want to save, file name will be taken by default.

        Examples
        --------
        >>> from tigerml.eda import EDAReport
        >>> import pandas as pd
        >>> df = pd.read_csv("titatic.csv")
        >>> an = EDAReport(df, y = 'Survived', y_continuous = False)
        >>> an.key_drivers(quick=False)
        """
        if y:
            assert isinstance(y, str) or isinstance(y, list)
            ys = y if isinstance(y, list) else [y]
            self._set_xy_cols(ys)
        else:
            ys = [self._current_y] if self._current_y else None
        if not ys:
            raise Exception("dependent variable name needs to be passed")
        key_drivers = {}
        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols
        for y in ys:
            self._set_current_y(y, self.y_continuous)
            self._preprocess_data(self._current_y)
            key_drivers[y] = {}
            key_drivers[y]["feature_scores"] = self.get_feature_scores(
                features=features, quick=quick
            )
            key_drivers[y]["feature_importances"] = self.feature_importances(
                features=features, quick=quick
            )
            # if not quick:
            # 	key_drivers[y]['feature_selection'] = {
            # 		'recursive_feature_elimination': self.feature_elimination_analysis()}
            key_drivers[y]["pca_analysis"] = self.get_pca_analysis(features=features)
            # if not quick:
            x_vars = list(set(features) & set(self.x_cols))
            top_n = None
            if len(x_vars) > 50:
                top_n = 50
            joint_plots = self.bivariate_plots(
                x_vars=x_vars, y_vars=self._current_y, return_dict=True, top_n=top_n
            )
            key_drivers[y]["bivariate_plots"] = joint_plots
        # key_drivers[y]['tsne_projection'] = self.get_tsne_projection()
        if save_as:
            default_report_name = "key_drivers_report_at_{}".format(time_now_readable())
            save_path = append_file_to_path(save_path, default_report_name + save_as)
            create_report(
                key_drivers,
                path=save_path,
                format=save_as,
                split_sheets=True,
            )
        self.key_drivers_report = key_drivers
        return key_drivers
