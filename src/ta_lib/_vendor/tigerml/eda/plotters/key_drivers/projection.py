import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .scoring import FeatureCorrelation


class PCAProjection:
    """
    The PCAProjection provides functionality for projecting a multi-dimensional.

    Dataset into either 2 or 3 components so they can be plotted as a scatter plot on
    2d or 3d axes.
    """

    def __init__(self):
        # fig, ax = plt.subplots()
        self.plots = []

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fits the visualizer on the input data, and returns transformed object.

        Parameters
        ----------
        X : DataFrame of shape n x m
          A data frame of n instances with m features where m>2.
        y : pd.Series of shape (n,), optional
          series with target values for each instance in X. This
          vector is used to determine the color of the points in X.

        Returns
        -------
        self :
            the modified object.
        """
        self.X = X
        pca_transformer = Pipeline(
            [("scale", StandardScaler(with_std=True)), ("pca", PCA(2))]
        )
        pca_transformer.fit(X)
        self.dims = pd.DataFrame(pca_transformer.transform(X))
        for ind, dim in enumerate(self.dims):
            viz = FeatureCorrelation(False)
            viz.fit(X, self.dims[dim])
            plot = viz.get_plot()
            self.plots.append(
                plot.opts(title="Feature Correlation with Dimension {}".format(ind + 1))
            )
        return self.dims

    def get_plots_data(self):
        """Returns plots data."""
        return self.dims

    def get_plots(self, type="bokeh"):
        """Takes input from '.fit' and returns the PCA projection plot for the data."""
        # self.finalize()
        return self.plots


# from yellowbrick.features import Manifold
# class TSNEProjection(Manifold):
#     """Tsne projection class."""
#
#     def __init__(self):
#         # fig, ax = plt.subplots()
#         self.plots = []
#         super().__init__(manifold="tsne")
#
#     def fit_transform(self, X, y=None, **kwargs):
#         """Returns transformed plot dimensions data."""
#         self.X = X
#         self.dims = pd.DataFrame(super().fit_transform(X, y, **kwargs))
#         for ind, dim in enumerate(self.dims):
#             viz = FeatureCorrelation(False)
#             viz.fit(X, self.dims[dim])
#             plot = viz.get_plot()
#             self.plots.append(
#                 plot.opts(title="Feature Correlation with Dimension {}".format(ind + 1))
#             )
#         return self.dims
#
#     def get_plots_data(self):
#         """Returns plots data."""
#         return self.dims
#
#     def get_plots(self, type="bokeh"):
#         """Returns plots."""
#         # self.finalize()
#         return self.plots
