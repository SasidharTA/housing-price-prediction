import numpy as np
import pandas as pd
from math import ceil


class TimeSeriesSplit:
    """
    Class to perform the split of time series data into train and test data either with cross validation or single split based on test percentage or cutoff dates.

    For future implementation of cross validation type,
    'rolling_train_test_split' should be considered

    For future implementation of split type,
    'n_points_type' should be considered

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    date_column : str
        The name of the column that represents the date.

    Examples
    --------
    >>> from tigerml.automl.backends.ts_algos.TimeSeriesSplit import (
            TimeSeriesSplit
        )
    >>> data_split = TimeSeriesSplit(
        data=df, date_column=date_col
        )
    >>> split_index = (
        data_split._split_without_cv(train_test_split_dict)
        )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str,
    ):
        self.data = data
        self.date_column = date_column

    def _split_with_test_perc(self, test_perc: float = None):
        """
        Split the data based on test percentage.

        Generate indices to split data into training and test set with
        the help of a test percentage

        Parameters
        ----------
        test_perc : float, optional
            The percentage of data to be used for testing, Defaults to 0.2.

        Returns
        -------
        [(train_idx, test_idx)] : A list of tuples where each tuple contains the indices
                of the train and test data similar to sklearn model selection.
        """

        test_perc = 0.2 if test_perc is None else test_perc

        data = self.data
        date_column = self.date_column

        sorted_index = [
            x
            for _, x in sorted(
                zip(np.array(data[date_column]), np.arange(0, len(data[date_column]))),
                reverse=False,
            )
        ]

        n_train = ceil((1 - test_perc) * len(data))

        train_idx = sorted_index[:n_train]
        test_idx = sorted_index[n_train:]

        return [(train_idx, test_idx)]

    def _split_with_cutoff_date(
        self, train_cutoff_dates: dict, test_cutoff_dates: dict
    ):
        """
        Split data based on cutoff dates.

        Generate indices to split data into training and test set with
        the help of a cutoff date.

        Parameters
        ----------
        train_cutoff_dates : dict
            A dictionary representing the start and end date for train data.
        test_cutoff_dates : dict
            A dictionary representing the start and end date for test data.

        Returns
        -------
        [(train_idx, test_idx)]: A list of tuples where each tuple contains the indices
            of the train and test data similar to sklearn model selection.
        """

        assert train_cutoff_dates is not None, "train_cutoff_dates supplied as None"
        assert test_cutoff_dates is not None, "test_cutoff_dates supplied as None"

        data = self.data
        date_column = self.date_column

        train_cutoff_start_date = pd.Timestamp(train_cutoff_dates["start_date"])
        train_cutoff_end_date = pd.Timestamp(train_cutoff_dates["end_date"])
        test_cutoff_start_date = pd.Timestamp(test_cutoff_dates["start_date"])
        test_cutoff_end_date = pd.Timestamp(test_cutoff_dates["end_date"])

        train_idx = (
            data[date_column]
            .loc[
                lambda x: (x >= train_cutoff_start_date) & (x <= train_cutoff_end_date)
            ]
            .index
        )
        test_idx = (
            data[date_column]
            .loc[lambda x: (x >= test_cutoff_start_date) & (x <= test_cutoff_end_date)]
            .index
        )

        return [(train_idx, test_idx)]

    def _split_without_cv(self, split_kwargs: dict):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        split_kwargs : dict
            A dictionary that specifies the splitting arguments.
            when split_kwargs["type"] == 'test_perc_type',
            split_kwargs['test_perc'], split_kwargs['date_column'] should exist
            else if  split_kwargs["type"] == 'cutoff_type',
            split_kwargs['train_cutoff_date'], split_kwargs['test_cutoff_date'],
            split_kwargs['date_column'] should exist

        Returns
        -------
        [(train_idx, test_idx)]: A list of tuples where each tuple contains the indices
                of the train and test data similar to sklearn model selection
        """

        split_type = split_kwargs["type"]
        split_kwargs_dict = split_kwargs[split_type]

        if split_type == "test_perc_type":
            index_output = self._split_with_test_perc(**split_kwargs_dict)

        elif split_type == "cutoff_type":
            index_output = self._split_with_cutoff_date(**split_kwargs_dict)
        else:
            index_output = [(self.data.index, self.data.index)]

        return index_output

    def _split_cross_validation(
        self,
        n_splits: int = None,
        gap: int = None,
        n_test_datapoints: int = None,
        stride: int = None,
    ):
        """
        Generate indices to split data into training and test set for cross validation.

        TODO:
        - Assertions on the cv_kwargs (keys, Data Type, Validate Values supplied)

        Parameters
        ----------
        n_splits : int, optional
            Number of splits to perform. Defaults to None.
        gap : int, optional
            Gap between training and testing data. Defaults to None.
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets.
        n_test_datapoints : int, optional
            number of time units to include in each test set, Defaults to None.
        stride : int, optional
            number of time units to be moved after each train-test split. Defaults to None.

        Returns
        -------
        [(train_idx, test_idx)]: A list of tuples where each tuple contains the indices
            of the train and test data similar to sklearn model selection.
        """

        date_column = self.date_column
        data = self.data

        gap = 0 if gap is None else gap
        n_test_datapoints = 4 if n_test_datapoints is None else n_test_datapoints
        stride = 1 if stride is None else stride

        initial_train_points = data.shape[0]
        possible_splits = ceil(
            ((initial_train_points - n_test_datapoints) - (n_test_datapoints + gap) + 1)
            / stride
        )

        n_splits_final = possible_splits if n_splits is None else n_splits

        counter = 0
        check_train_test_ratio = True

        cv_list = list()

        n_train_points_for_split = initial_train_points

        sorted_index = [
            x
            for _, x in sorted(
                zip(np.array(data[date_column]), np.arange(0, len(data[date_column]))),
                reverse=False,
            )
        ]

        while (counter <= n_splits_final) & (check_train_test_ratio):

            n_train_points_temp = n_train_points_for_split - n_test_datapoints - gap

            check_train_test_ratio = (n_train_points_temp / n_test_datapoints) >= 1

            if check_train_test_ratio:
                cv_list.append(
                    (
                        sorted_index[:n_train_points_temp],
                        sorted_index[
                            (n_train_points_temp + gap) : (n_train_points_for_split)
                        ],
                    )
                )
                counter += 1
                n_train_points_for_split = n_train_points_for_split - stride

        if len(cv_list) == 0:
            cv_list.append((sorted_index, sorted_index))

        return cv_list

    def split(self, main_split_kwargs: dict):
        """
        Generate indices to split data into training and test set indices / index based on whether cross validation is required or not.

        Parameters
        ----------
        main_split_kwargs : dict[dict]
            when main_split_kwargs['split_cv'] == False,
                split_kwargs = main_split_kwargs['split_kwargs'],
                for more info refer to the methods
                '_split_with_test_perc' & '_split_with_cutoff_date'

            else if main_split_kwargs['split_cv'] == True,
                cv_kwargs = main_split_kwargs['cv_kwargs']
                for more info in this aspect, refer to the method
                '_split_cross_validation'

        """

        split = main_split_kwargs["split"]

        if split == "cv":
            cv_kwargs = main_split_kwargs[split]
            index_output = self._split_cross_validation(**cv_kwargs)
        else:
            split_kwargs = main_split_kwargs[split]
            index_output = self._split_without_cv(split_kwargs)

        return index_output
