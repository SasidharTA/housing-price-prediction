import numpy as np
from sklearn.metrics.pairwise import euclidean_distances  # noqa
from sklearn.preprocessing import LabelEncoder

DIAMETER_METHODS = ["mean_cluster", "farthest"]
CLUSTER_DISTANCE_METHODS = ["nearest", "farthest"]


def inter_cluster_distances(labels, distances, method="nearest"):
    """Calculate the distances between the two nearest points or the two farthest points for each cluster, depending on the method specified.

    The input `distances` should be a symmetric matrix. This function only considers the upper triangle of the matrix (i.e., the distances above the main diagonal). If a distance matrix is not given, Euclidean distance is used.
    If `method='nearest'`, this function returns the distances between the two nearest points in each cluster.
    If `method='farthest'`, this function returns the distances between the two farthest points in each cluster.

    Parameters
    ----------
    labels : list
             A list containing cluster labels for each of the n elements
    distances : numpy.array
                An n x n numpy.array containing the pairwise distances between elements
    method : str, optional (default='nearest')
             The method to use for calculating inter-cluster distances.
             'nearest' calculates the distances between the two nearest points in each cluster,
             while 'farthest' calculates the distances between the two farthest points.

    Returns
    -------
    cluster_distances : A numpy.array of inter-cluster distances.
    """
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError("method must be one of {}".format(CLUSTER_DISTANCE_METHODS))

    if method == "nearest":
        return __cluster_distances_by_points(labels, distances)
    elif method == "farthest":
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full(
        (n_unique_labels, n_unique_labels), float("inf") if not farthest else 0
    )

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (
                    not farthest
                    and distances[i, ii] < cluster_distances[labels[i], labels[ii]]
                )
                or (
                    farthest
                    and distances[i, ii] > cluster_distances[labels[i], labels[ii]]
                )
            ):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]
                ] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method="farthest"):
    """Calculate diameter for clusters.

    Parameters
    ----------
    labels : a list containing cluster labels for each of the n elements

    distances : an n x n numpy.array containing the pairwise distances between elements

    method : either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other

    Returns
    -------
    diameters :

    """
    if method not in DIAMETER_METHODS:
        raise ValueError("method must be one of {}".format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == "mean_cluster":
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == "farthest":
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method="farthest", cdist_method="nearest"):
    """Dunn index for cluster validation (larger is better).

    min_inter_distance: The minimum distance between any two cluster centroids.
    max_intra_distance: The maximum diameter (intra-cluster distance) among all the clusters.
    To calculate the min_inter_distance, you compute the pairwise distance between the centroids of different clusters and take the minimum value.
    To calculate the max_intra_distance, you find the diameter (maximum distance) within each cluster and take the maximum value among all the clusters.

    dunn Index = min_inter_distance / max_intra_distance

    Parameters
    ----------
    labels : a list containing cluster labels for each of the n elements

    distances : an n x n numpy.array containing the pairwise distances between elements

    diameter_method : see :py:function:`diameter` `method` parameter

    cdist_method: see : py:function:`diameter` `method` parameter


    """

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter
