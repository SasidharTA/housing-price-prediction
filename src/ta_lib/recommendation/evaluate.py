import numpy as np
import pandas as pd

__all__ = ["mean_average_precision_at_k"]


def avg_precision_k(relevant_item_set, predicted_item_set):
    """Compute the average precision at k between 2 lists of items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    relvant_flags = np.isin(predicted_item_set, relevant_item_set)
    precisions_at_k = np.array([])
    count_relevant = 0
    for i, val in enumerate(relvant_flags):
        if val:
            count_relevant += 1
        precisions_at_k = np.append(precisions_at_k, count_relevant / (i + 1))

    avg_precision_at_k = (
        sum(precisions_at_k * relvant_flags) / relevant_item_set.shape[0]
    )
    return avg_precision_at_k


def mean_average_precision_at_k(relevant_item_set, predicted_item_set, k=5):
    """Compute the mean average precision at k between two list of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    num_users = len(relevant_item_set)
    total_sum_precision = 0
    for i in range(num_users):
        relevant = np.array(relevant_item_set[i])
        predicted = np.array(predicted_item_set[i][:k])
        total_sum_precision += avg_precision_k(relevant, predicted)
    return total_sum_precision / num_users


def train_test_split(
    df, user_column, samples=5, random_state=7, date=None, date_val=None, fraction=0.3
):
    def generate_samples(x):
        size = x.shape[0]
        if size == 1:
            return None
        if size < samples:
            return x.sample(frac=fraction, random_state=random_state)
        else:
            return x.sample(n=samples, random_state=random_state)

    grouped_data = df.groupby(user_column)

    if date is None:
        test = grouped_data.apply(generate_samples)
        test.index = [i[1] for i in test.index]
    else:
        test = df[df[date] > date_val]

    train = df.loc[~df.index.isin(test.index), :]

    # keeping common users accross train and test dataset
    common_users = list(set(train[user_column]) & set(test[user_column]))
    train = train[train[user_column].isin(common_users)]
    test = test[test[user_column].isin(common_users)]

    return (train, test)


def generate_util_matrix(df, user_col, pivot_cols, values_col, aggfunc="count"):
    """Generate UTIL matrix that will be used for cosine similarity.

    Parameters
    ----------
    df : list of dataframes pd.DataFrame
        Dataframe that needs to be converted to pivot table
    pivot_cols : list[]
        column names to be pivoted
    index_cols : list[]
        column name to be converted to index of pivot table

    """
    pivot_df = None
    if type(pivot_cols) == list and len(pivot_cols) > 1:
        for i in pivot_cols:
            sub_df = df.pivot_table(
                index=user_col,
                columns=i,
                values=values_col,
                aggfunc=aggfunc,
            ).fillna(0)
            if pivot_df is None:
                pivot_df = sub_df
            else:
                pivot_df = pd.concat([pivot_df, sub_df], axis=1)
                # pd.merge(pivot_df, sub_df, on=index_cols)

    else:
        pivot_df = df.pivot_table(
            index=user_col, columns=pivot_cols, values=values_col, aggfunc="count"
        ).fillna(0)
    return pivot_df


def generate_train_test_util_matrix(
    datasets, user_col, pivot_cols, values_col, aggfunc="count"
):
    """
    Generate UTIL matrix and split the data into train and test set.

    Parameters
    ----------
    datasets : list of dataframes pd.DataFrame
        Dataframe that needs to be converted to pivot table
    pivot_cols : list[]
        column names to be pivoted
    user_col : list[]
        column name to be converted to index of pivot table
    values_col : string
        name of the feedback column which needs to be used

    """
    full_col_list = []
    for i in pivot_cols:
        full_col_list += list(datasets[0][i])
        full_col_list += list(datasets[1][i])
    full_col_list = list(set(full_col_list))
    train_util = generate_util_matrix(
        datasets[0], user_col, pivot_cols, values_col, aggfunc
    )
    test_util = generate_util_matrix(
        datasets[0], user_col, pivot_cols, values_col, aggfunc
    )
    train_util[list(set(full_col_list) - set(train_util.columns))] = 0
    test_util[list(set(full_col_list) - set(test_util.columns))] = 0
    return train_util, test_util
