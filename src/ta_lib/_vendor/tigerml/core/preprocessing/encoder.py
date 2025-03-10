# from bunch import Bunch
import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from tigerml.core.utils import DictObject, compute_if_dask

_LOGGER = logging.getLogger(__name__)


class Encoder:
    """Encoder class."""

    METHODS = DictObject({"onehot": "onehot", "ordinal": "ordinal", "target": "target"})

    MAX_BUCKETS = 20
    MIN_SAMPLE_PERC = 0.02
    MIN_SAMPLE_ABS = 20
    LONG_TAIL_CHECK = 0.1

    def __init__(self, data, y=None):
        if not data.__module__.startswith("tigerml"):
            from tigerml.core.dataframe.helpers import tigerify

            data = tigerify(data)
        data = data.categorize()
        self.data = data
        self.encoding_rules = []
        self.encoded_data = self.data
        self.encoding_summary = {}
        self.encoding_mapping = {}
        self.y = None
        if y:
            self.y = self.data[y]
            self.data = self.data.drop(y, axis=1)

    def add_encode_rule(self, cols, method, **kwargs):
        """Adds rule dictionary to encoding rules."""
        cols_already_applied = [
            col for col in cols if any([col in rule[0] for rule in self.encoding_rules])
        ]
        if cols_already_applied:
            raise Exception(
                "Encoding rule already applied " "for {}".format(cols_already_applied)
            )
        if [col for col in cols if col not in self.data]:
            raise Exception(
                "Columns not present in "
                "data - {}".format([col for col in cols if col not in self.data])
            )
        if method not in self.METHODS:
            raise Exception(
                "Supported imputation methods: " "{}".format(self.METHODS.keys())
            )
        rule_dict = {"cols": cols, "method": method}
        rule_dict.update(kwargs)
        self.encoding_rules.append(rule_dict)
        return self

    def _default_encoding(self, encode_columns):
        _LOGGER.info(
            "Encoding categorical variables with default settings which will "
            "not be ideal. "
            "Processing these variables manually is highly recommended."
        )
        # encode_columns = [col for col in non_numeric_columns if col in cols]
        if self.y is not None:
            self.encoding_rules.append(
                {"cols": encode_columns, "method": self.METHODS.target}
            )
            for col in encode_columns:
                min_samples = (
                    round(self.MIN_SAMPLE_PERC * len(self.data))
                    if self.MIN_SAMPLE_PERC
                    else self.MIN_SAMPLE_ABS
                )
                levels_with_less_min_for_target = [
                    x
                    for x in compute_if_dask(self.data[col].unique())
                    if len(self.data[self.data[col] == x]) < min_samples
                ]
                if levels_with_less_min_for_target:
                    _LOGGER.info(
                        "{} has levels with less than {}{} values. "
                        "Target encoding in such cases is not "
                        "recommended.".format(
                            col,
                            min_samples,
                            f" ({self.MIN_SAMPLE_PERC*100}%)"
                            if self.MIN_SAMPLE_PERC
                            else "",
                        )
                    )
        else:
            for col in encode_columns:
                num_of_levels = self.data[col].nunique()
                if num_of_levels <= self.MAX_BUCKETS:
                    self.encoding_rules.append(
                        {"cols": col, "method": self.METHODS.onehot}
                    )
                else:
                    min_samples = (
                        (self.MIN_SAMPLE_PERC * len(self.data))
                        if self.MIN_SAMPLE_PERC
                        else self.MIN_SAMPLE_ABS
                    )
                    buckets_with_min_samples = [
                        x
                        for x in compute_if_dask(self.data[col].unique())
                        if len(self.data[self.data[col] == x]) >= min_samples
                    ]
                    if (
                        len(buckets_with_min_samples)
                        > num_of_levels * self.LONG_TAIL_CHECK
                    ):
                        groups = buckets_with_min_samples
                        self.encoding_rules.append(
                            {
                                "cols": col,
                                "method": self.METHODS.onehot,
                                "groups": groups,
                            }
                        )
                    else:
                        _LOGGER.info(
                            "CANNOT ENCODE {}. A good encoding "
                            "method is not found.".format(col)
                        )
                        continue

    def transform(self, cols=[]):
        """Returns encoded data after transformation."""
        if not cols:
            from tigerml.core.utils import get_dt_cols, get_num_cols

            numeric_columns = get_num_cols(self.data) + get_dt_cols(self.data)
            cols = [col for col in self.data.columns if col not in numeric_columns]
            cols_set = [
                col
                for col in cols
                if not (
                    any(
                        [
                            "encoded_{}".format(col) in data_col
                            for data_col in self.data.columns
                        ]
                    )
                )
            ]
            if len(cols_set) < len(cols):
                _LOGGER.info(
                    "Encoding {} columns. Columns - {} are "
                    "already encoded.".format(
                        len(cols_set), list(set(cols) - set(cols_set))
                    )
                )
        else:
            cols_set = cols
        self.encoded_data = self.data
        if not self.encoding_rules:
            self._default_encoding(cols_set)
        for rule in self.encoding_rules:
            cols = rule.pop("cols")
            if isinstance(cols, str):
                cols = [cols]
            cols = [col for col in cols if col in cols_set]
            method = rule.pop("method")
            kwargs = rule.copy()
            if method == self.METHODS.target and "target" not in kwargs:
                if self.y is None:
                    raise Exception("Need target for target encoding")
                else:
                    kwargs.update({"target": self.y})
            for col in cols:
                if method == self.METHODS.onehot:
                    encoded = self.onehotEncode(self.data[col], **kwargs)
                    self.encoded_data = td.concat([self.encoded_data, encoded], axis=1)
                elif method == self.METHODS.ordinal:
                    encoded, mapper = self.ordinalEncode(self.data[col], **kwargs)
                    self.encoding_mapping.update({col: mapper})
                    if encoded.name in self.encoded_data:
                        _LOGGER.info(
                            "{} already exists in data. "
                            "Overriding it.".format(encoded.name)
                        )
                        self.encoded_data[encoded.name] = encoded
                    else:
                        self.encoded_data = td.concat(
                            [self.encoded_data, encoded], axis=1
                        )
                elif method == self.METHODS.target:
                    encoded, mapper = self.targetEncode(self.data[col], **kwargs)
                    encoded.index = self.encoded_data.index
                    self.encoding_mapping.update({col: mapper})
                    if encoded.name in self.encoded_data.columns:
                        _LOGGER.info(
                            "{} already exists in data. "
                            "Overriding it.".format(encoded.name)
                        )
                        self.encoded_data[encoded.name] = encoded
                    else:
                        self.encoded_data = td.concat(
                            [self.encoded_data, encoded], axis=1
                        )
                self.encoding_summary.update(
                    {
                        col: {
                            "original_type": self.data.dtypes.astype(str)[col],
                            "new_columns": encoded.columns
                            if hasattr(encoded, "columns")
                            else encoded.name,
                            "method": "{} encoded".format(method),
                        }
                    }
                )
        if self.y is not None:
            # import pdb
            # pdb.set_trace()
            self.encoded_data = td.concat([self.encoded_data, self.y], axis=1)
        return self.encoded_data

    def get_encoding_method(self, col_name):
        """Gets encoding method."""
        if [rule for rule in self.encoding_rules if col_name in rule["cols"]]:
            return [rule for rule in self.encoding_rules if col_name in rule["cols"]][
                0
            ]["method"]
        else:
            return None

    @staticmethod
    def onehotEncode(feature, prefix="onehot_encoded", **kwargs):
        """
        This method one hot encodes all the factors in category variable.

        Parameters
        ----------
          feature : str
            Name of the category variable to be encoded.
          prefix : str
            Default is 'OHE'. The prefex will be appended to encoded variable.
            Ex: 'OHE_VariableName_FactorName'

        Returns
        -------
          dataframe :
            Modified dataframe will be returned.

        """
        prefix = prefix + "_" + feature.name
        if feature.isna().sum() > 0:
            include_na = True
        else:
            include_na = False
        dummies = td.get_dummies(feature, dummy_na=include_na)
        dummies = dummies.rename(
            columns=dict(
                zip(
                    list(dummies.columns),
                    [(prefix + "_" + str(x)) for x in dummies.columns],
                )
            )
        )
        if "groups" in kwargs:
            new_dummies = td.DataFrame(backend=feature.backend)
            for group in kwargs["groups"]:
                group_name = prefix + "_" + str(group)
                if isinstance(group, str):
                    new_dummies[prefix + "_" + group] = dummies[group_name]
                elif isinstance(group, list):
                    if len(group) == 1:
                        group = group[0]
                        new_dummies[prefix + "_" + group] = dummies[group_name]
                    else:
                        dummy_name = "grouped_{}".format("_".join(group))
                        new_dummies[dummy_name] = dummies[group_name].sum()
                else:
                    raise Exception("Incorrect input for groups")
                dummies = dummies.drop(group_name, axis=1)
            if not dummies.empty:
                new_dummies[prefix + "_other"] = dummies.sum()
            dummies = new_dummies
        return dummies

    @staticmethod
    def ordinalEncode(feature, mapper, prefix="ordinal_encoded"):
        """
        This method ordinally encodes all the factors in category variable.

        Parameters
        ----------
          feature : str
            Name of the category variable to be encoded.
          mapper : dict
            Dictionary with factor to encoding value mapping.
            Ex: If the variable has following levels low, medium and high and
            you want to ordinal encode them
            use the following mapper.
            mapper = {'low':1, 'medium':2, 'high':3}
          prefix : str
            Default is 'ORB'. The prefex will be appended to encoded variable.
            Ex: 'ORB_VariableName_FactorName'
        Returns
        -------
          dataframe :
            Modified dataframe will be returned.

        """
        encoded_name = prefix + "_" + feature.name
        encoded = feature.map(mapper)
        encoded.rename(encoded_name)
        if encoded.isnull().sum() > 0:
            _LOGGER.info(
                "Few levels are missing in the mapper, "
                "appended such records with nans"
            )
        return encoded, mapper

    @staticmethod
    def targetEncode(
        feature, target, min_samples=1, smoothing=1, prefix="target_encoded"
    ):
        """Target Encode.

        This transformation is applied on categorical variable for a regression task.
        Each factor value is replaced by the average of the response variable within
        the factor group.

        Parameters
        ----------
          feature : str
            Name of the category variable to be encoded.
          min_samples : int
            Default is 1. Min no of samples required within each factor.
          smoothing : int
            Default is 1. Smoothens variation in the transformation by giving
            more weight to the prior average.
          prefix : str
            Default is 'TGE'. The prefex will be appended to encoded variable.
            Ex: 'TGE_VariableName_FactorName'
        Returns
        -------
          dataframe :
            Modified dataframe will be returned.
        """
        encoded_name = prefix + "_" + feature.name
        df = td.concat([feature, target], axis=1)
        averages = df.groupby(by=feature.name)[target.name].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples) / smoothing))
        prior = df.loc[:, target.name].mean()
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * (smoothing)
        averages = averages.drop(["mean", "count"], axis=1)
        encoded = td.merge(
            td.DataFrame(feature),
            td.DataFrame(
                averages.reset_index().rename(
                    columns={"index": feature.name, target.name: "average"}
                )
            ),
            on=feature.name,
            how="left",
        )["average"].fillna(prior)
        encoded = encoded.rename(encoded_name)
        return encoded, averages.rename(columns={target.name: "encoded values"})
