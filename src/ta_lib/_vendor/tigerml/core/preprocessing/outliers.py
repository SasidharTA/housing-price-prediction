class Outlier:
    """Transformer to treat outliers by either capping them or dropping them.

    Parameters
    ----------
    method: str, default is percentile.
        Accepted values are mean, median, percentile and threshold.
        The method that is used to determine the upper/lower bounds.

            * If percentile - bounds are quantiles (0-1).
                Default bounds are (0.01, 0.99)
            * If mean - bounds are no. of standard deviations from mean.
                Default bounds are (3, 3)
            * If median - bounds are no. of IQRs away from median.
                Default bounds are  (1.5, 1.5)
            * If threshold - Fixed values for outlier identification
                should be provided through lb and/or ub variables. No default bounds.
    lb: dict, default is none
        lb is the lower bound
        * If not None, pass a dictionary of columns
        with custom lower limits for each
    ub: dict, default is none
        ub is the upper bound
        * If not None, pass a dictionary of columns
        with custom upper limits for each
    """

    def __init__(
        self,
        method="percentile",
        lb=None,
        ub=None,
    ):
        """Initialize Estimator."""
        if method not in ["mean", "median", "percentile", "threshold"]:
            raise ValueError(
                "Unsupported outlier method '{}', should be "
                "one of from ['mean', 'median', 'percentile', "
                "'threshold']".format(method)
            )
        default_limits = {
            "percentile": (0.01, 0.99),
            "mean": (3, 3),
            "median": (1.5, 1.5),
            "threshold": ({}, {}),
        }
        self.method = method

        if self.method == "threshold" and lb is None and ub is None:
            raise ValueError(
                "For 'threshold' method, a dictionary is required "
                "for variable 'lb' and/or 'ub', "
                "having columns as keys and custom outlier "
                "thresholds defined for each column as values."
            )
        if lb is None:
            lb = default_limits[method][0]
        if ub is None:
            ub = default_limits[method][1]

        self.lb = lb
        self.ub = ub

    def fit(self, X, cols=None):
        """Compute outlier limits from X.

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independent features
        cols : list, optional
            List of column names for features relevant when
            X is Arrays, by default None

        """
        (self.lb, self.ub) = self._compute_outlier_bounds(
            X, cols, self.method, self.lb, self.ub
        )
        return self

    def transform(self, X, drop=False):
        """Treat outliers for X.

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independent features.
        drop: bool, default is False
            If True, records contains outlier will be dropped.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe is returned
        """
        df = X.copy()
        for (col, lb) in self.lb.items():
            fil_lower = df[col] < lb
            if drop is True:
                df = df[~fil_lower]
            else:
                df.loc[fil_lower, col] = lb
        for (col, ub) in self.ub.items():
            fil_upper = df[col] > ub
            if drop is True:
                df = df[~fil_upper]
            else:
                df.loc[fil_upper, col] = ub
        return df

    def fit_transform(
        self,
        X,
        drop=False,
        cols=None,
    ):
        """Fit to data, then transform it.

        Parameters
        ----------
        X: pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independent features
        drop: bool, default is False
            If True, records contains outlier will be dropped.
        cols: list, optional
            List of column names of features, by default None

        Returns
        -------
        pd.DataFrame
            Transformed dataframe is returned
        """
        return self.fit(X, cols).transform(X, drop)

    def _compute_outlier_bounds(  # noqa
        self,
        df,
        cols=None,
        method="percentile",
        lb=None,
        ub=None,
    ):
        """Compute outlier bounds for each column.

        Parameters
        ----------
        df: pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independent features.
        cols: list, optional
            List of column names of features, by default None
        method: str, default is percentile
            Accepted values are mean, median, threshold, percentile
        lb: numeric, default is none
            lb is the lower bound
            * If not None, pass a dictionary of columns with lower limits for each
        ub: numeric, default is none
            ub is the upper bound
            * If not None, pass a dictionary of columns with upper limits for each

        Returns
        -------
        tuple
            a tuple of dictionaries for lb and ub values of all columns.
        """
        if cols is None:
            cols = df.select_dtypes("number").columns.to_list()

        num_df = df[cols]
        if method == "mean":
            mean = num_df.mean()
            std = num_df.std()
            lb = (mean - lb * std).to_dict()
            ub = (mean + ub * std).to_dict()
        elif method == "median":
            fst_quant = num_df.quantile(0.25)
            thrd_quant = num_df.quantile(0.75)
            iqr = thrd_quant - fst_quant
            lb = (fst_quant - lb * iqr).to_dict()
            ub = (thrd_quant + ub * iqr).to_dict()
        elif method == "percentile":
            lb = num_df.quantile(lb).to_dict()
            ub = num_df.quantile(ub).to_dict()
        elif method == "threshold":
            pass
        else:
            raise ValueError("Unsupported outlier method : " + method)

        return (lb, ub)

    def get_outlier_nums(self, df):
        """Return a dictionary containing outlier numbers for each columns in df.

        Parameters
        ----------
        df: pd.DataFrame
            Data for which outlier number information require.

        Returns
        -------
        outlier_nums: pd.DataFrame
        """
        outlier_nums = {}
        for col in df.columns:
            outlier_nums[col] = [0, 0]
        for (col, lb) in self.lb.items():
            outlier_nums[col][0] = (df[col] < lb).sum()
        for (col, ub) in self.ub.items():
            outlier_nums[col][1] = (df[col] > ub).sum()
        return outlier_nums

    def get_outlier_indexes(self, df):
        """Return a dictionary containing outlier indexes for each columns in df.

        Parameters
        ----------
        df: pd.DataFrame
            Data for which outlier index information is required.

        Returns
        -------
        outlier_indexes: dict
            key -- column name
            value -- list of outlier indexes
        """
        outlier_indexes = {}
        for col in df.columns:
            outlier_indexes[col] = list([])
        for (col, lb) in self.lb.items():
            outlier_list = df.index[df[col] < lb].tolist()
            if outlier_list is not None and len(outlier_list) > 0:
                outlier_indexes[col].extend(df.index[df[col] < lb].tolist())
        for (col, ub) in self.ub.items():
            outlier_list = df.index[df[col] > ub].tolist()
            if outlier_list is not None and len(outlier_list) > 0:
                outlier_indexes[col].extend(df.index[df[col] > ub].tolist())
        return outlier_indexes
