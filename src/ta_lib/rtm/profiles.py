from posixpath import pardir  # noqa

import pandas as pd


def cluster_profiles(df, min_val=0.7, max_val=1.3):
    """Calculate cluster profiles and return them as a styled pandas dataframe.

    Compute cluster profiles by calculating the mean of each feature for each cluster, divided by the overall mean of that feature across all clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing the cluster labels and data to be used for computing the profiles.
    min_val : float, optional
        Minimum threshold value for styling the output dataframe. Default is 0.7.
    max_val : float, optional
        Maximum threshold value for styling the output dataframe. Default is 1.3.

    Returns
    -------
    cluster_profile : pandas.io.formats.style.Styler
                      Styled pandas dataframe containing the computed cluster profiles.

    Examples
    --------
    >>> data = pd.DataFrame({'Label': [0, 0, 1, 1], 'x': [1, 2, 3, 4], 'y': [4, 5, 6, 7]})
    >>> cluster_profiles_int(data)

    """
    lab_dt = df
    cp = dict()
    for j in range(len(lab_dt["Label"].unique())):
        cp[f"cluster{j}"] = (
            lab_dt[lab_dt["Label"] == j].iloc[:, :-1].mean()
            / lab_dt.iloc[:, :-1].mean()
        )
    cluster_profile = pd.DataFrame([cp[i] for i in list(cp.keys())]).T
    cluster_profile.columns = [
        "cluster_" + str(i) for i in list(cluster_profile.columns)
    ]
    cluster_profile = cluster_profile.style.apply(
        lambda x: [
            "background-color: #f55f64"
            if v < min_val
            else "" + "background-color: #4ced61"
            if v > max_val
            else ""
            for v in x
        ],
        axis=1,
    )
    # cluster_profile.to_excel(writer,engine='openpyxl')
    return cluster_profile


def cluster_profiles_int(df, min_val=0.7, max_val=1.3):
    """Calculate cluster profiles and return them as a styled pandas dataframe.

    Compute the cluster profiles for a given dataframe by calculating the mean of each feature for each cluster, as well as the overall mean of each feature.

    Parameters
    ----------
    df : pandas.DataFrame
         Input data containing the cluster labels and data to be used for computing the profiles.
    min_val : float, optional
              Minimum threshold value for styling the output dataframe. Default is 0.7.
    max_val : float, optional
              Maximum threshold value for styling the output dataframe. Default is 1.3.

    Returns
    -------
    cluster_profile : pandas.io.formats.style.Styler
                      Styled pandas dataframe containing the computed cluster profiles.

    Examples
    --------
    >>> data = pd.DataFrame({'Label': [0, 0, 1, 1], 'x': [1, 2, 3, 4], 'y': [4, 5, 6, 7]})
    >>> cluster_profiles_int(data)

    """
    lab_dt = df
    cp = dict()
    cpo = dict()  # noqa
    for j in range(len(lab_dt["Label"].unique())):
        cp[f"cluster{j}"] = lab_dt[lab_dt["Label"] == j].iloc[:, :-1].mean()
    cp["overall"] = lab_dt.iloc[:, :-1].mean()
    cluster_profile = pd.DataFrame([cp[i] for i in list(cp.keys())]).T
    col_names = ["cluster_" + str(i) for i in list(cluster_profile.columns[:-1])]
    col_names.append("overall_mean")
    cluster_profile.columns = col_names
    cluster_profile = cluster_profile.style.apply(
        lambda x: [
            "background-color: #f55f64"
            if v / x[-1] < min_val
            else "" + "background-color: #4ced61"
            if v / x[-1] > max_val
            else ""
            for v in x
        ],
        axis=1,
    )
    return cluster_profile
