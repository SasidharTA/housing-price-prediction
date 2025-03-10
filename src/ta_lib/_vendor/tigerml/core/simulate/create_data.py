"""Utility functions to create data."""

import logging
import numpy as np
import pandas as pd
import random
from datetime import timedelta
from itertools import cycle
from scipy.stats import poisson, skewnorm
from sklearn import datasets

_LOGGER = logging.getLogger(__name__)


def generate_string(n):
    """Generate random string of length n.

    Parameters
    ----------
    n: int
        length of string to be generate

    Returns
    -------
    result_str: str
    """
    characters = "abcdefghijklmnoprstuvwxyz"
    result_str = "".join(random.choice(characters) for i in range(n))
    return result_str


def numerical_distribution_samples(var_dist, n_samples, random_state):
    """Return random number array based on the distribution specified in var_dist.

    Parameters
    ----------
    var_dist: dict
        distribution info for a numerical variable
    n_samples: int
        number of samples
    random_state: int
        random state for random sampling

    Returns
    -------
    y: np.array
    """
    if var_dist is None or "type" not in var_dist.keys():
        var_dist = {"type": "normal", "a": 0, "loc": 100, "scale": 5}
    if var_dist["type"] == "normal":
        y = skewnorm.rvs(
            a=var_dist["a"],
            loc=var_dist["loc"],
            scale=var_dist["scale"],
            size=n_samples,
            random_state=random_state,
        )
    elif var_dist["type"] == "poisson":
        y = poisson.rvs(
            mu=var_dist["mu"],
            loc=var_dist["loc"],
            size=n_samples,
            random_state=random_state,
        )
    elif var_dist["type"] == "custom":
        y = var_dist["func"](**var_dist["kwargs"])
    else:
        raise ValueError(
            "Please pass distribution type as one "
            "of from 'normal', 'poisson' or 'custom'!"
        )
    return y


def categorical_distribution_samples(
    var_dist, n_samples, generate_random_strings=False
):
    """Return random class array based on the distribution specified in var_dist.

    Parameters
    ----------
    var_dist: dict
        distribution info for a categorical variable
    n_samples: int
        number of samples
    generate_random_strings: bool
        whether to generate class names through random strings or not

    Returns
    -------
    y: np.array
    """
    if var_dist["type"] in ["probs", "multinomial"]:
        if "class_names" in var_dist.keys():
            classes = var_dist["class_names"]
        elif generate_random_strings:
            classes = [generate_string(5) for i in range(len(var_dist["probs"]))]
        else:
            classes = np.arange(len(var_dist["probs"]))
        if np.sum(var_dist["probs"]) != 1:
            class_probs = var_dist["probs"] / np.sum(var_dist["probs"])
            _LOGGER.warning(
                "probabilities {} do not sum to 1, "
                "so values will be normalized as {}.".format(
                    var_dist["probs"], class_probs
                )
            )
    if var_dist["type"] == "probs":
        y = np.random.choice(classes, size=n_samples, p=var_dist["probs"])
    elif var_dist["type"] == "multinomial":
        vals = np.random.multinomial(n=n_samples, pvals=var_dist["probs"])
        probs = vals / n_samples
        y = np.random.choice(classes, size=n_samples, p=probs)
    elif var_dist["type"] == "custom":
        y = var_dist["func"](**var_dist["kwargs"])
    else:
        raise ValueError(
            "Please pass distribution type as one of "
            "from 'probs', 'multinomial' or 'custom'!"
        )
    return y


def get_num_X_samples(n_samples, n_features_num, num_X_dist, random_state):
    """Generate numerical X based on custom distribution specified in num_X_dist for all n_features_num.

    Parameters
    ----------
    n_samples: int
        number of samples
    n_features_num: int
        number of features
    num_X_dist: list of dict
        custom distribution info for all 'n_features_num' numeric features
    random_state: int
        random state for random sampling

    Returns
    -------
    X: np.array
    """
    x = np.zeros((n_samples, n_features_num))
    if len(num_X_dist) == n_features_num:
        for i, dist in enumerate(num_X_dist):
            random_state = (
                random_state + 1
            )  # for getting different values for different variables
            x_cap = numerical_distribution_samples(dist, n_samples, random_state)
            x[:, i] = x_cap
    else:
        raise ValueError(
            "No of features (n_features_num) and list of "
            "numerical distributions (num_X_dist) do not match"
        )
    return x


def get_cat_X_samples(n_samples, cat_X_dist, random_state):
    """Generate categorical X based on custom distribution specified in cat_X_dist.

    Parameters
    ----------
    n_samples: int
        number of samples
    cat_X_dist: list of dict
        custom distribution info for categorical features,
        each dict will corresponds to a single categorical variable
    random_state: int
        random state for random sampling

    Returns
    -------
    X1: pandas.DataFrame
        dataframe with categorical features
    """
    X1 = pd.DataFrame()
    for i, var_dist in enumerate(cat_X_dist):
        random_state = random_state + 1
        random.seed(random_state)
        np.random.seed(random_state)
        X1[f"cat_feat_{i + 1}"] = categorical_distribution_samples(
            var_dist, n_samples, generate_random_strings=True
        )
        X1[f"cat_feat_{i + 1}"] = X1[f"cat_feat_{i + 1}"].astype("category", copy=False)
    return X1


def random_date(start, end, frequency="d"):
    """Return a random datetime between two datetime objects.

    Parameters
    ----------
    start: datetime
        starting date
    end: datetime
        ending date
    frequency: str, default='d'
        frequency for generating random date,
        one of from 'd' (Days), 'h' (Hours), 'm' (Minutes) or 's' (Seconds)

    Returns
    -------
    date: datetime
    """
    delta = end - start
    if frequency == "d":
        int_delta = delta.days
        random_day = random.randrange(int_delta)
        output = start + timedelta(days=random_day)
    elif frequency == "h":
        int_delta = delta.total_seconds() // 3600
        random_hour = random.randrange(int_delta)
        output = start + timedelta(hours=random_hour)
    elif frequency == "m":
        int_delta = delta.total_seconds() // 60
        random_minute = random.randrange(int_delta)
        output = start + timedelta(minutes=random_minute)
    elif frequency == "s":
        int_delta = delta.total_seconds()
        random_second = random.randrange(int_delta)
        output = start + timedelta(seconds=random_second)
    else:
        raise ValueError(
            "Invalid frequency value '{}', should be one "
            "of 'd' (Days), 'h' (Hours), 'm' (Minutes) or "
            "'s' (Seconds).".format(frequency)
        )
    return output


def create_data(
    n_samples: int,
    n_features_num: int,
    is_classification: bool,
    n_classes=2,
    target_as_func=True,
    perc_missing=0,
    perc_zeros=0,
    num_X_dist=None,
    cat_X_dist=None,
    target_dist=None,
    date_cols=None,
    include_id_column=False,
    include_id_column_unique=False,
    random_state=1234,
    **kwargs,
):
    """
    Generate Sample dataset with required set of features.

    Parameters
    ----------
    n_samples : int
        Total number of samples.

    n_features_num : int
        Total number of numerical features.

    is_classification : bool
        True for classification and False for regression problems.

    n_classes : int, optional (default=2)
        The number of classes, applicable only for classification problem.

    target_as_func : bool, optional (default=True)
        Whether the target should be a function of X or can be
        picked randomly from distribution samples.

    perc_missing : float, optional (default=0)
        Percentage of missing values in each of the columns.
        Value is between 0 and 1.

    perc_zeros : float, optional (default=0)
        Percentage of zeroes in numerical values.
        Value is between 0 and 1.

    num_X_dist : list of dict, optional (default=None)
        Custom distribution info for numerical X.

        Note: length of ``num_X_dist`` must be equals to ``n_features_num``.

        For each numeric variable (``n_features_num``), pass a dictionary
        with required parameters according to distribution type as follows:

        - For normal/gaussian dist: expected args for ``scipy.stats.skewnorm.rvs()``

          e.g. ``{"type": "normal", "a":0, "loc": 100, "scale": 5}``

        - For poisson dist: expected args for scipy.stats.poisson.rvs()``

          e.g. ``{"type": "poisson", "mu":0, "loc": 100}``

        - For custom dist: pass an appropriate function under 'func' key
          and expected arguments under 'kwargs' key. e.g.

          .. code-block:: python

            from tweedie import tweedie

            def func(mu, p, phi, n_samples):
                return tweedie(mu=mu, p=p, phi=phi).rvs(n_samples)

            {"type": "custom", "func":func,
             "kwargs": {'mu':100, 'p':1.5, 'phi':20, 'n_samples':n_samples}}

          Note: Here, 'n_samples' value must be equals to ``n_samples`` variable passed.

        Note: If empty dictionary is passed, then default value
        ``{"type": "normal", "a":0, "loc": 100, "scale": 5}`` will
        be consider for that variable. This will be same in case 'type'
        key itself is missing from the dictionary.

    cat_X_dist : list of dict, optional (default=None)
        Custom distribution info for categorical X.

        Each dictionary would correspond to generate a single categorical column.
        For each categorical variable, pass type of distribution under key 'type',
        optional class names under key 'class_names', and other expected arguments
        based on the distribution.

        Currently, the following types are supported:

        - 'multinomial': multinomial distribution defined by ``np.random.multinomial()``

           e.g.

           .. code-block:: python

                {'type':'multinomial',
                 'probs':[0.3, 0.4, 0.2, 0.1],
                 'class_names':['abc', 'xyz', 'pqr', 'ijk']}

        - 'probs': uniform/non-uniform probabilities distribution defined by
          ``np.random.choice()``

          e.g. ``{'type':'probs', 'probs':[0.3, 0.4, 0.3]}``

        - 'custom': user can pass a direct function which should generate samples
          of size ``n_samples``. The dictionary should be of the form
          ``{'type':'custom', 'func':func, 'kwargs':{}}``

          e.g. ``{'type':'multinomial', 'probs':[0.3, 0.4, 0.2, 0.1]}``,
          ``{'type':'probs', 'probs':[0.3, 0.4, 0.3]}`` would correspond
          to generate 2 categorical columns having 4 and 3 classes
          (i.e. length of 'probs') respectively.

        Note: In 'multinomial' and 'probs' types:

          - Sum of 'probs' values should be 1 otherwise values will
            be normalized internally to make the sum as 1.

          - Optional class names can be passed under key "class_names",
            if passed will use the same otherwise will generate 5 characters
            random string for each category corresponds to probability
            specified in 'probs'.

    target_dist : dict, optional (default=None)
        Distribution of target variable (y) to draw the random samples.

        For regression, it is used when ``target_as_func=False`` and for
        classification, used when ``target_as_func=False`` and ``num_X_dist``
        is not None.

        For regression, pass distribution info for y as expected similar
        for a variable in ``num_X_dist``. Default value is
        ``{"type": "normal", "a":0, "loc": 100, "scale": 5}``.

        For classification, pass distribution info for y as expected similar
        for a variable in ``cat_X_dist``.Default value is
        ``{'type':'probs', 'probs':[1/n_classes]*n_classes, 'class_names':np.arrange(n_classes)}``

        Note: For 'probs' and 'multinomial' type, the sum of 'probs' values
        should be 1, otherwise values will be normalized internally.

    date_cols : list of dict, optional (default=None)
        Pass a list of dictionaries, each corresponding to a date column.

        Each dictionary would contain 'start' and 'end' date values
        in "%Y-%m-%d" format along with 'frequency' value to pick
        random dates within the same.

        Default value of 'start', 'end' and 'frequency'
        will be considered as "1990-01-01", "today", and 'd' respectively
        in case when respective value is missing from the dictionary.

        Allowed values for 'frequency' are one of from 'd' (Days),
        'h' (Hours), 'm' (Minutes), and 's' (Seconds).

        For example,
        ``{'start':"2018-10-23", 'end':"2020-10-30", 'frequency':'s'}``,
        ``{'start':"2016-08-01", 'end':"2017-07-30", 'frequency':'d'}``
        would indicate generating 2 date columns which will have random date
        at specified 'frequency' level between 'start' and 'end' dates
        for the individual.

    include_id_column : bool, optional (default=False)
        If an Id column to be included in the dataset.

    include_id_column_unique : bool, optional (default=False)
        If a unique Id column to be added to the dataset.

    random_state : int, optional (default=1234)
        Random state for random sampling.

    **kwargs : acceptable keyword arguments
        - If ``target_as_func=True``, then pass the valid kwargs for
          sklearn's ``make_classification()`` or ``make_regression()``.
        - Else, if ``num_X_dist=None``, then pass the valid kwargs for
          sklearn's ``make_gaussian_quantiles()``.
        - Else, not required.

    Returns
    -------
    pandas.DataFrame
        Synthetic dataset with the required set of features.

    """
    # seed the random state for repetition of same data
    random.seed(random_state)
    np.random.seed(random_state)

    # generate the categorical data
    X1 = pd.DataFrame()
    if cat_X_dist is not None:
        X1 = get_cat_X_samples(n_samples, cat_X_dist, random_state)

    # generate the numerical data
    if target_as_func:
        # if y should be derived as a function of features from X
        if is_classification:
            X2, y = datasets.make_classification(
                n_samples=n_samples,
                n_features=n_features_num,
                n_classes=n_classes,
                random_state=random_state,
                **kwargs,
            )
        else:
            X2, y = datasets.make_regression(
                n_samples=n_samples,
                n_features=n_features_num,
                random_state=random_state,
                **kwargs,
            )
    elif num_X_dist is not None:
        # when custom distribution info passed for X and/or y
        X2 = get_num_X_samples(n_samples, n_features_num, num_X_dist, random_state)
        if not is_classification:
            y = numerical_distribution_samples(target_dist, n_samples, random_state)
        else:
            if n_classes != len(target_dist["probs"]):
                raise ValueError(
                    "target_dist['probs'] size must be equals to n_classes!"
                )
            y = categorical_distribution_samples(
                target_dist, n_samples, generate_random_strings=False
            )
    else:
        # if no info passed for custom distribution of X
        X2, y = datasets.make_gaussian_quantiles(
            n_samples=n_samples,
            n_features=n_features_num,
            n_classes=n_classes,
            random_state=random_state,
            **kwargs,
        )
        if not is_classification:
            y = numerical_distribution_samples(target_dist, n_samples, random_state)

    X2 = pd.DataFrame(X2)
    X2.columns = [f"num_feat_{i + 1}" for i in range(X2.shape[1])]

    # join categorical if available with numerical data
    df = pd.concat([X1, X2], axis=1)

    # if perc_missing value is specified, for each feature
    # we impute NAN for that percent of samples
    if perc_missing > 0:
        missing_vals = int(n_samples * perc_missing)
        for i, col in enumerate(df.columns):
            indices = random.sample(range(n_samples), missing_vals)
            df.iloc[indices, i] = np.nan

    # if perc_zeros value is specified, for each numerical feature
    # we impute 0 for that percent of samples
    if perc_zeros > 0:
        zero_vals = int(n_samples * perc_zeros)
        num_cols = df.select_dtypes("number").columns
        for i, col in enumerate(num_cols):
            indices = random.sample(range(n_samples), zero_vals)
            df.loc[indices, col] = 0

    # define the target variable
    df["target"] = y

    # build a relationship of target (y) with categorical variables
    # if not is_classification:
    # # Todo: for regression - understand and evaluate the
    #                          current logic of relationship building
    #     str_cols = list(set(df.columns) - set(df.select_dtypes('number')))
    #     for col in str_cols:
    #         cats_ = df[col].unique().tolist()
    #         for cat in cats_:
    #             fil_ = (df[col] == cat)
    #             df.loc[fil_, 'target'] = df.loc[fil_, 'target'] +
    #             random.randint(0, len(cats_)) * fil_.astype(int)
    # else:
    # # Todo: for classification - implement a logic of relationship building
    #     pass

    # include date column(s)
    if date_cols is not None:
        for i, x in enumerate(date_cols):
            if "start" in x.keys():
                start_date = pd.to_datetime(x["start"], format="%Y-%m-%d")
            else:
                start_date = pd.to_datetime("1990-01-01", format="%Y-%m-%d")
            if "end" in x.keys():
                end_date = pd.to_datetime(x["end"], format="%Y-%m-%d")
            else:
                end_date = pd.to_datetime("today", format="%Y-%m-%d")
            if "frequency" in x.keys():
                freq = x["frequency"]
                if freq not in ["d", "h", "m", "s"]:
                    raise ValueError(
                        "Invalid frequency value '{}', should be "
                        "one of 'd' (Days), 'h' (Hours), "
                        "'m' (Minutes) or 's' (Seconds).".format(freq)
                    )
            else:
                freq = "d"
            df["date_feat_" + str(i + 1)] = [
                random_date(start_date, end_date, freq) for i in range(n_samples)
            ]

    # include a id column
    if include_id_column:
        if include_id_column_unique:
            id_ = np.arange(n_samples)
        else:
            id_ = [random.randint(0, n_samples) for i in range(n_samples)]
        df["ID"] = id_
    return df
