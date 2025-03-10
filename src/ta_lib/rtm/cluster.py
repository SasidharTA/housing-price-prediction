import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from ta_lib.rtm.dunn import dunn  # noqa


def pca_plot(dataframe, label, reverse=False, palette="coolwarm"):
    """Generate a scatter plot of the first two principal components of the input dataframe .

    The `pca_plot()` function uses PCA to compute the first two principal components of the input dataframe, and generates a scatter plot of the principal components colored by cluster label.
    The `reverse` parameter allows the user to reverse the order of the principal components in the plot.
    The resulting plot shows how well the observations in the input dataframe cluster in principal component space.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input data containing the features for which to compute principal components.
    label : pandas.Series
        Series containing the cluster labels for each observation in the input dataframe.
    reverse : bool, optional
        If True, reverse the order of the principal components in the plot. Default is False.
    palette : str, optional
        Color palette to use for plotting. Default is 'coolwarm'.

    Examples
    --------
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> label = pd.Series([0, 1, 0])
    >>> pca_plot(data, label)
    """
    plt.figure(figsize=(5, 5))
    pc = PCA(n_components=2)
    pca_df = pd.DataFrame(
        pc.fit_transform(dataframe),
        columns=["PC" + str(i) for i in range(pc.n_components_)],
    )
    if reverse:
        sns.scatterplot(x="PC1", y="PC0", hue=label, data=pca_df, palette=palette)
    else:
        sns.scatterplot(x="PC0", y="PC1", hue=label, data=pca_df, palette=palette)
    i = max(label) + 1
    plt.ylabel(f"PCA plot for {i} clusters")
    plt.xlabel("")
    plt.legend(loc="right", title="No of clusters")


class KMeansClustering:
    """Class allows to get the cluster metrics by building clusters on KMeansClustering++ algorithm.

    It takes a dataframe and the min and max range of number clusters you want to try.

    Parameters
    ----------
    df (DataFrame) :  A pandas DataFrame that contains the data to be clustered.
    low (int) :  Lower limit of number of clusters to be tried. Default value is 2.
    high (int) : Upper limit of number of clusters to be tried. Default value is 8.

    Examples
    --------
    >>> data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    >>> df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
    >>> km = KMeansClustering(df, low=2, high=6)
    >>> km.fit(method='pca', pc_var=0.95)
    >>> km.elbow_plot()
    >>> km.silhoutte_plot()
    >>> km.pcaplot()
    >>> km.silhoutte_analysis_plot()
    >>> labelled_data = km.get_labelled_data(4)
    """

    def __init__(self, df, low=2, high=8):
        self.low = low
        self.high = high
        self.df = df
        self.METHODS = ["scaled", "pca"]

    def fit(self, method="pca", pc_var=0.95):
        """Fits the KMeansClustering clustering model with the specified method.

        This method fits a KMeans clustering model on the data stored in the `self.df` attribute.
        The method used for clustering is specified with the `method` parameter,
        which can be either 'pca' or 'scaled'.
        For each number of clusters in the range from `self.low` to `self.high`,
        the sum of squared distances of samples to their closest cluster center (inertia),
        predicted labels for each sample,
        and the silhouette score of the clustering are calculated and stored in the `inertia`,
        `predicted`, and `sc_score` instance variables, respectively.
        A labelled dataframe is also created for each number of clusters and stored in the `labelled_df` instance variable. The labelled dataframe contains the original data plus a column indicating the predicted cluster label for each sample.

        Parameters
        ----------
        method : str, optional
        The method used for clustering. Must be one of {'pca', 'scaled'} (default is 'pca').
        pc_var : float, optional
        The minimum variance of PCA to get at least two components (default is 0.95)."
        """
        if method not in self.METHODS:
            raise ValueError("method must be one of {}".format(self.METHODS))
        self.inertia = []
        self.predicted = []
        self.sc_score = []
        self.dunn_val = []

        pca = PCA(n_components=1)
        pca.fit_transform(self.df)
        min_var = round(sum(pca.explained_variance_ratio_), 2)
        if pc_var <= min_var:
            raise ValueError(
                "variance of pca should be more than {} to get atleast two components".format(
                    min_var
                )
            )

        pc = PCA(n_components=pc_var)
        self.pca_fd = pd.DataFrame(
            pc.fit_transform(self.df),
            columns=["PC" + str(i) for i in range(pc.n_components_)],
        )
        if method == "pca":
            d = euclidean_distances(self.pca_fd)
            for i in range(self.low, self.high + 1):
                self.cluster = KMeans(n_clusters=i, random_state=42).fit(self.pca_fd)
                k = self.cluster.predict(self.pca_fd)
                self.inertia.append(self.cluster.inertia_)
                self.predicted.append(k)
                self.sc_score.append(
                    silhouette_score(self.pca_fd, self.cluster.labels_)
                )
                # self.dunn_val.append(dunn(k, d))
        if method == "scaled":
            d = euclidean_distances(self.df)  # noqa
            for i in range(self.low, self.high + 1):
                self.cluster = KMeans(n_clusters=i, random_state=42).fit(self.df)
                k = self.cluster.predict(self.df)
                self.inertia.append(self.cluster.inertia_)
                self.predicted.append(k)
                self.sc_score.append(silhouette_score(self.df, self.cluster.labels_))
                # self.dunn_val.append(dunn(k, d))
        self.labelled_df = dict()
        j = 0
        for i in range(self.low, self.high + 1):
            self.labelled_df[i] = pd.concat(
                [self.df, pd.DataFrame(self.predicted[j], columns=["Label"])], axis=1
            )
            j = j + 1

    def elbow_plot(self):
        """Generate elbow plot for the KMeansClustering model.

        Generates a plot of the sum of squared distances of samples to their closest cluster center (inertia) versus the number of clusters.
        The range of number of clusters is determined by the `low` and `high` attributes of the KMeans object.
        """
        plt.plot(range(self.low, self.high + 1), self.inertia)
        plt.title("Elbow plot", fontsize="large", fontweight="bold", color="#091463")
        plt.xlabel("No of Clusters", fontfamily="sans-serif", fontweight="bold")
        plt.ylabel("Inertia", fontfamily="sans-serif", fontweight="bold")

    def silhoutte_plot(self):
        """Generate silhouette score plot for the KMeansClustering model.

        Generates a plot of the silhouette score versus the number of clusters. The range of number of clusters is determined by the `low` and `high` attributes of the KMeans object. The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters, with values ranging from -1 to 1.
        The higher the score, the better the clustering.
        """
        plt.plot(range(self.low, self.high + 1), self.sc_score)
        plt.title(
            ("Plot using Silhoutte score"),
            fontsize="large",
            fontweight="bold",
            color="#091463",
        )
        plt.xlabel("No of Clusters", fontfamily="sans-serif", fontweight="bold")
        plt.ylabel("Silhoutte Score", fontfamily="sans-serif", fontweight="bold")

    def dunn_plot(self):
        """Generate dunn index plot for the KMeansClustering model.

        Generates a plot of the dunn index versus the number of clusters. The range of number of clusters is determined by the `low` and `high` attributes of the KMeans object. The dunn index is a measure of the clustering quality, defined as the ratio of the minimum inter-cluster distance to the maximum intra-cluster distance. The higher the dunn index, the better the clustering.
        """
        plt.plot(range(self.low, self.high + 1), self.dunn_val)
        plt.title(
            ("Plot using dunn score"),
            fontsize="large",
            fontweight="bold",
            color="#091463",
        )
        plt.xlabel("No of Clusters", fontfamily="sans-serif", fontweight="bold")
        plt.ylabel("dunn Index Value", fontfamily="sans-serif", fontweight="bold")

    def pcaplot(self):
        """Generate scatter plot of the data in the PCA space for each number of clusters.

        Generates a scatter plot of the data in the PCA space for each number of clusters in the range from `self.low` to `self.high`. The predicted cluster labels for each sample are used to color code the data points. The scatter plot is generated using the `pca_plot` function, which takes two arguments: the data to be plotted and the predicted cluster labels for each sample.

        """
        length = len(range(self.low, self.high + 1))
        for i in range(length):
            pca_plot(self.df, self.predicted[i])

    def silhoutte_analysis_plot(self):
        """Generate silhouette analysis plot for the KMeansClustering model.

        Generates a plot of the silhouette coefficients for each sample in each cluster, organized by the number of clusters. The range of number of clusters is determined by the `low` and `high` attributes of the KMeans object. The plot shows the silhouette coefficient values for each sample, with a red dashed line indicating the average silhouette score of all the values.
        The silhouette coefficient is a measure of how similar an object is to its own cluster compared to other clusters, with values ranging from -1 to 1. The higher the score, the better the clustering. The plot is generated using the `silhouette_samples` and `silhouette_score` functions from the `sklearn.metrics` module.
        """
        j = 0
        for n_clusters in range(self.low, self.high + 1):
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)
            X = self.pca_fd
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            cluster_labels = self.predicted[j]
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[
                    cluster_labels == i
                ]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            j = j + 1

    def get_lablled_data(self, cluster_no):
        """Return the labelled data for a specific number of clusters.

        The labelled data is a pandas DataFrame that contains the original data along with a column of predicted cluster labels for each sample. The DataFrame is created for each number of clusters in the range from `self.low` to `self.high` when the `fit` method is called. Use this method to retrieve the labelled data for a specific number of clusters.

        Parameters
        ----------
        cluster_no : int
                    The number of clusters for which the labelled data should be returned.

        Returns
        -------
        self.labelled_df[cluster_no] : pandas.DataFrame
                                      The labelled data for the specified number of clusters."
        """
        return self.labelled_df[cluster_no]
