import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
import warnings
from pandas.api.types import infer_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from tigerml.core.common import compute_correlations
from tigerml.core.preprocessing.outliers import Outlier
from tigerml.core.utils import measure_time
from tigerml.core.utils.constants import SUMMARY_KEY_MAP
from tigerml.core.utils.pandas import (
    get_bool_cols,
    get_cat_cols,
    get_dt_cols,
    get_non_num_cols,
    get_num_cols,
    is_discrete,
)
from tigerml.core.utils.segmented import (
    calculate_all_segments,
    get_segment_filter,
)

from .encoder import Encoder
from .imputer import Imputer

_LOGGER = logging.getLogger(__name__)

"""
def is_discrete(series_data):
    from sklearn.utils.multiclass import type_of_target
    from tigerml.core.dataframe import convert_to_tiger_assets

    val = convert_to_tiger_assets(type_of_target(series_data[series_data.notnull()]))
    if "continuous" not in val:
        return True
    else:
        return False
"""


def get_mixed_dtypes(data):
    """Get columns of a DataFrame having mixed data types."""
    if isinstance(data, (pd.DataFrame, td.DataFrame)):
        dtypes = data.apply(infer_dtype)
    else:
        dtypes = pd.Series([infer_dtype(data)])
    if len(dtypes):
        mixed_dtypes = dtypes[dtypes.str.startswith("mixed")].to_dict()
    else:
        mixed_dtypes = dict()
    return mixed_dtypes


def clean_mixed_dtypes(data):
    """Clean DataFrame having mixed data types by deleting such columns."""
    mixed_dtypes = get_mixed_dtypes(data)
    if len(mixed_dtypes):
        message = "Some columns in the data have mixed dtypes "
        message += f"which need to be cleaned/dropped: {mixed_dtypes}."
        data_cleaned = data.copy()
        cleaned_cols = []
        for col, dtype in mixed_dtypes.items():
            if dtype == "mixed-integer-float":
                # In case of mixed integer and float type,
                # convert all values to float
                data_cleaned[col] = data[col].astype(float)
                cleaned_cols.append(col)
        if cleaned_cols:
            for col in cleaned_cols:
                del mixed_dtypes[col]
        dropped_cols = list(mixed_dtypes.keys())
        data_cleaned = data_cleaned.drop(dropped_cols, axis=1)
        message += f"\nDropped columns: {dropped_cols}" if dropped_cols else ""
        message += f"\nCleaned columns: {cleaned_cols}" if cleaned_cols else ""
        warnings.warn(message)
        return data_cleaned
    else:
        return data


class DataProcessor:
    """Data Processor class."""

    def __init__(self, data, segment_by=None, y=None, y_continuous=None):
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        self.data = data
        self.data = self._clean_mixed_dtypes()
        self.segment_by = None
        self.all_segments = None
        if segment_by:
            self.segment_by = (
                segment_by if isinstance(segment_by, list) else [segment_by]
            )
            self._calculate_all_segments()
        self.is_classification = None
        self.original_columns = list(self.data.columns)
        self.processed_data = self.data
        self._current_y = None
        self._y_for_encoding = None
        self.is_cleaned = False
        self.is_encoded = False
        self.y_cols = []
        self.x_cols = list(self.data.columns)
        self.num_cols = self.get_numeric_columns()
        self.cat_cols = self.get_cat_columns()
        self.dt_cols = self.get_dt_columns()
        self.processing_summary = {}
        self.cleaning_summary = {}
        self.y_continuous = y_continuous
        if y is not None:
            self._set_xy_cols(y, y_continuous)

    def _clean_mixed_dtypes(self):
        return clean_mixed_dtypes(self.data)

    def _get_segment_filter(self, segment):
        return get_segment_filter(self.data, self.segment_by, segment)

    def _calculate_all_segments(self):
        self.all_segments = calculate_all_segments(self.data, self.segment_by)

    def _check_nulls_in_y_cols(self, y_cols):
        for y in y_cols:
            if self.data[y].isnull().sum() > 0:
                _LOGGER.warning(
                    "Dependent variable '{}' having {} null values "
                    "and the same will be treated in the "
                    "subsequent step of EDA process.".format(
                        y, self.data[y].isnull().sum()
                    )
                )

    def _set_y_cols(self, y=None):
        # data = self.processed_data.reset_index(drop=True)
        if isinstance(y, str):
            y = [y]
        if y is not None:
            self._check_nulls_in_y_cols(y)
            if self.get_non_numeric_columns(self.data[y]):
                _LOGGER.exception(
                    "Dependent variable cannot be non numeric. "
                    "Please process the data."
                )
                raise Exception(
                    "Dependent variable cannot be non numeric. "
                    "Please process the data."
                )
            return y
        else:
            return []

    def _set_current_y(self, y, y_continuous=None):
        self._current_y = y
        if y_continuous:
            self.is_classification = False
        else:
            # self.is_classification = self._is_discrete(self.data[self.y_cols[0]])
            self.is_classification = is_discrete(self.data[self.y_cols[0]])
            if len(self.y_cols) > 1:
                _LOGGER.warning(
                    "Multiple dependant variables are defined. "
                    "y_continuous is not passed. Inferring as {}".format(
                        "discrete" if self.is_classification else "continuous"
                    )
                )

    def _set_xy_cols(self, y, y_continuous=None):
        self.y_cols = self._set_y_cols(y=y)
        self.x_cols = self._get_x_cols()
        self._set_current_y(self.y_cols[0], y_continuous)

    def _get_x_cols(self, y=None):
        # y = self._current_y if self._current_y else None
        data = self.data
        if y is not None:
            y = [y] if isinstance(y, str) else y
        else:
            y = self.y_cols
        if self.y_cols:
            return [col for col in data.columns if col not in y]
        else:
            return list(data.columns)

    def get_numeric_columns(self, data=None):
        """List the numeric columns in dataframe."""
        if data is None:
            data = self.data
        return get_num_cols(data)

    def get_non_numeric_columns(self, data=None):
        """List the non-numeric columns in dataframe."""
        if data is None:
            data = self.data
        return get_non_num_cols(data)

    def get_dt_columns(self, data=None):
        """List the date columns in dataframe."""
        if data is None:
            data = self.data
        return get_dt_cols(data)

    def get_cat_columns(self, data=None):
        """List the categorical columns in dataframe."""
        if data is None:
            data = self.data
        return get_cat_cols(data)

    def get_bool_columns(self, data=None):
        """List the bool columns in dataframe."""
        if data is None:
            data = self.data
        return get_bool_cols(data)

    def _impute_na(self, df=None, impute_num_na=None, impute_cat_na=None, segment=True):
        if df is None:
            df = self.data
        if self.segment_by and segment:
            imputed_data = td.DataFrame()
            for segment in self.all_segments:
                current_imputed = self._impute_na(
                    df[self._get_segment_filter(segment)],
                    impute_num_na,
                    impute_cat_na,
                    segment=False,
                )
                imputed_data = td.concat([imputed_data, current_imputed])
            return imputed_data.sort_index()
        imputer = Imputer()
        if df.isna().sum().sum() > 0:
            _LOGGER.info(
                "Imputing missing values with median for "
                "numeric data and mode for categorical data."
            )
            if impute_num_na:
                imputer.num_impute_method = impute_num_na
            if impute_cat_na:
                imputer.cat_impute_method = impute_cat_na
            # print(df.columns)
            df = imputer.fit_transform(df)
            df = pd.DataFrame(df, columns=imputer.get_feature_names())
            df = df.infer_objects()
            df = td.DataFrame(df)
            imputation_summary = pd.DataFrame(imputer.imputation_summary_).T
            imputation_summary = imputation_summary[
                imputation_summary["no_of_missing"] != 0
            ]
            self.cleaning_summary["missing"] = imputation_summary
        else:
            self.cleaning_summary["missing"] = "No missing values in data"
            _LOGGER.info("No missing values")
        return df

    # def add_imputation_rule(self, *args, **kwargs):
    #     return self.imputer.add_imputation_rule(*args, **kwargs)

    def _drop_inf(self, df):
        if df.isin([np.Inf]).sum().sum() > 0:
            _LOGGER.info("Dropping infinity values")
            pre_length = len(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            post_length = len(df)
            self.cleaning_summary[
                "infinitys"
            ] = "Dropped {} rows due to infinity values".format(
                pre_length - post_length
            )
            return df
        else:
            self.cleaning_summary["infinitys"] = "No Infinity values"
            _LOGGER.info("No Infinity values")
            return df

    def _handle_outliers(self, method="mean", lb=None, ub=None, drop=False, cols=None):
        df = self.data
        if not cols:
            cols = [col for col in get_num_cols(df) if col not in get_bool_cols(df)]
        else:
            cols = [
                col
                for col in cols
                if col in get_num_cols(df) and col not in get_bool_cols(df)
            ]
        if method is not None:
            df = Outlier(method=method, lb=lb, ub=ub).fit_transform(
                X=self.data, drop=drop, cols=cols
            )
        return df

    def _encode_non_numeric(self, y=None, segment=None):
        data = self.data
        if self.segment_by and not segment and y is not None:
            encoded_data = td.DataFrame()
            for segment in self.all_segments:
                current_encoded = self._encode_non_numeric(y=y, segment=segment)
                encoded_data = td.concat([encoded_data, current_encoded])
            return encoded_data.sort_index()
        if segment:
            data = data[self._get_segment_filter(segment)]
        self.encoding_summary = {}
        if self.get_bool_columns():
            _LOGGER.info(
                "Have {} boolean variables. Converting False to"
                " 0 and True to 1".format(len(self.get_bool_columns()))
            )
            for col in self.get_bool_columns():
                data[col] = data[col] * 1
                self.encoding_summary.update(
                    {
                        col: {
                            "original_type": "bool",
                            "new_columns": col,
                            "method": "True = 1, False = 0",
                        }
                    }
                )
        if not self.get_cat_columns():
            _LOGGER.info("No categorical variables in the data")
            return data
        if (y is None and self.is_encoded) or (
            y is not None and y == self._y_for_encoding
        ):
            return data
        # if y is None:
        #     y = self._current_y
        original_columns = list(set(self.original_columns) & set(data.columns))
        encoder = Encoder(data[original_columns], y=y)
        encoded_data = encoder.transform()
        self.encoding_summary.update(encoder.encoding_summary)
        self.processing_summary["encoding"] = pd.DataFrame(self.encoding_summary).T
        if encoder.encoding_mapping:
            if segment:
                pass
            else:
                self.processing_summary["encoded_mappings"] = encoder.encoding_mapping
        if not segment:
            self.is_encoded = True
            self._y_for_encoding = y
        return encoded_data

    def _clean_data(
        self,
        drop_na=False,
        impute_num_na=None,
        impute_cat_na=None,
        inf_is_outlier=False,
        outlier_method="mean",
        drop_outliers=False,
        outlier_lb_cols_dict=None,
        outlier_ub_cols_dict=None,
        outlier_cols=None,
    ):
        _LOGGER.info("Cleaning data")
        self.cleaning_summary = {}
        # df = self.data
        df = self._handle_outliers(
            method=outlier_method,
            drop=drop_outliers,
            lb=outlier_lb_cols_dict,
            ub=outlier_ub_cols_dict,
            cols=outlier_cols,
        )
        no_of_rows = len(df)
        if drop_na:
            df = df.dropna()
            message = "Dropped {} rows with missing values.".format(
                no_of_rows - len(df)
            )
            _LOGGER.info(message)
            self.cleaning_summary["missing"] = message
        else:
            df = self._impute_na(df, impute_num_na, impute_cat_na)
        if not inf_is_outlier:
            self.data = self._drop_inf(df)
        # df = self._handle_outliers(remove_outliers=None)
        self.data = df.reset_index(drop=True)
        self.is_cleaned = True
        self.processing_summary["imputations"] = self.cleaning_summary

    @measure_time(_LOGGER)
    def _preprocess_data(self, y=None):
        if not self.is_cleaned:
            outlier_cols = None
            if y:
                outlier_cols = [col for col in self.data.columns if col != y]
            self._clean_data(outlier_cols=outlier_cols)
        self.data = self._encode_non_numeric(y=y)

    def _apply_func(self, func, cols=None, axis=1, segment=True):
        df = self.data
        if cols:
            df = df[cols]
        if self.segment_by and segment:
            final_data = td.DataFrame()
            from tigerml.core.dataframe.helpers import is_series

            for segment in self.all_segments:
                current_computed = df[self._get_segment_filter(segment)].apply(
                    func, axis=axis
                )
                if is_series(current_computed):
                    current_computed = td.DataFrame(current_computed)
                for idx, seg_name in enumerate(self.segment_by):
                    current_computed[seg_name] = segment[idx]
                final_data = td.concat([final_data, current_computed])
            if not final_data.index.name:
                final_data.index.name = "tigerml_current_index"
            final_data = final_data.reset_index().set_index(
                self.segment_by + [final_data.index.name]
            )
            if len(final_data.columns) == 1:
                final_data = final_data[final_data.columns[0]]
            return final_data
        else:
            return df.apply(func, axis=axis)

    def _filter_cols_by_corr(self, threshold=0.25):
        imp_cols = []
        corr_values = None
        for y in self.y_cols:
            self._set_current_y(y)
            self._preprocess_data(self._current_y)
            corrs = compute_correlations(
                self.data,
                x_vars=[x for x in self.data.columns if x != self._current_y],
                y_vars=[self._current_y],
            )
            mergeable_corrs = corrs.drop(
                [SUMMARY_KEY_MAP.abs_corr_coef, SUMMARY_KEY_MAP.variable_2], axis=1
            ).rename(
                columns={SUMMARY_KEY_MAP.corr_coef: "Correlation with {}".format(y)}
            )
            if corr_values is not None:
                corr_values = corr_values.merge(
                    mergeable_corrs, on=SUMMARY_KEY_MAP.variable_1
                )
            else:
                corr_values = mergeable_corrs
            corrs = corrs[corrs[SUMMARY_KEY_MAP.abs_corr_coef] >= threshold]
            imp_cols += corrs[SUMMARY_KEY_MAP.variable_1].values.tolist()
        corr_values = corr_values.rename(
            columns={SUMMARY_KEY_MAP.variable_1: "Variable Name"}
        )
        return list(set(imp_cols)) + self.y_cols, corr_values

    def _prepare_data(self, corr_threshold):
        # import pdb
        # pdb.set_trace()
        prep_summary = {}
        constant_cols = [
            col for col in self.original_columns if self.data[col].nunique() == 1
        ]
        drop_cols = []
        elim_summary = {}

        if constant_cols:
            drop_cols += constant_cols
            elim_summary["summary"] = [
                "Dropped {} columns - {} having a constant value".format(
                    len(drop_cols), drop_cols
                )
            ]
            _LOGGER.info(
                "Dropped {} columns - {} having a constant value".format(
                    len(drop_cols), drop_cols
                )
            )

        if self.y_cols and corr_threshold is not None:
            imp_cols, corr_values = self._filter_cols_by_corr(corr_threshold)
            _LOGGER.info(
                "Selected {} features having correlation more than "
                "{} with {} column(s)".format(
                    len(imp_cols), corr_threshold, self.y_cols
                )
            )
            new_drop_cols = list(
                set(self.get_numeric_columns()) - set(drop_cols) - set(imp_cols)
            )

            if new_drop_cols:
                drop_cols += new_drop_cols
                message = [
                    "Dropped {} features having correlation less than "
                    "{} with {} column(s)".format(
                        len(new_drop_cols), corr_threshold, self.y_cols
                    )
                ]
                _LOGGER.info(
                    "Dropped {} features having correlation less than "
                    "{} with {} column(s)".format(
                        len(new_drop_cols), corr_threshold, self.y_cols
                    )
                )
                if "summary" in elim_summary:
                    elim_summary["summary"] += message
                else:
                    elim_summary["summary"] = message
                elim_summary["selected_features"] = corr_values[
                    corr_values["Variable Name"].isin(imp_cols)
                ].reset_index(drop=True)
                elim_summary["dropped_features"] = corr_values[
                    -corr_values["Variable Name"].isin(imp_cols)
                ].reset_index(drop=True)
                prep_summary["feature_elimination"] = elim_summary

        if drop_cols:
            self.data.drop(drop_cols, axis=1, inplace=True)
        self.x_cols = list(set(self.x_cols) - set(drop_cols))
        self.y_cols = list(set(self.y_cols) - set(drop_cols))
        self._preprocess_data(self._current_y)
        prep_summary.update(self.processing_summary)
        return prep_summary


def prep_data(
    data,
    dv_name,
    train_size=0.75,
    random_state=None,
    outlier_method="mean",
    drop_outliers=False,
    outlier_lb_cols_dict=None,
    outlier_ub_cols_dict=None,
    outlier_cols=None,
    remove_na=True,
    impute_num_na="",
    impute_cat_na="",
    stratify=False,
):
    """Prepare the raw data and return train and test subsets as required for modeling.

    Parameters
    ----------
    data: pd.DataFrame
        raw data
    dv_name: str
        dependent/target variable name
    train_size: float
        fraction of train data size wrt overall data
    random_state: int
        random state value for sampling
    outlier_method: str
        outlier method, one of from 'mean', 'median',
        'percentile' and 'threshold'
    drop_outliers:bool
        whether to drop records having outlier values
    outlier_lb_cols_dict: dict
        lower bound outlier limit, If not None, pass a dictionary
        of columns with custom lower limits for each
    outlier_ub_cols_dict: dict
        upper bound outlier limit, If not None, pass a dictionary
        of columns with custom upper limits for each
    outlier_cols: list
        column names list to be considered for outlier analysis
    remove_na: bool
        whether to drop records having null values
    impute_num_na: str
        imputation method for numerical columns, one of from "mean",
        "median", "mode", "constant" and "regression"
    impute_cat_na: str
        imputation method for categorical columns, one of from "mean",
        "median", "mode", "constant" and "regression"
    stratify: bool
        whether split the data in a stratified fashion, using dv_name
        as the class labels.

    Returns
    -------
    splitting : list
        List containing train-test split of data.
    """
    # warnings.filterwarnings('ignore')
    data = data.drop_duplicates()
    dp = DataProcessor(data, y=dv_name)
    dp._clean_data(
        drop_na=remove_na,
        impute_num_na=impute_num_na,
        impute_cat_na=impute_cat_na,
        outlier_method=outlier_method,
        drop_outliers=drop_outliers,
        outlier_lb_cols_dict=outlier_lb_cols_dict,
        outlier_ub_cols_dict=outlier_ub_cols_dict,
        outlier_cols=outlier_cols,
    )
    dp._encode_non_numeric(y=dv_name)
    data = dp.data[dp.get_numeric_columns()]
    target = data[[dv_name]]
    data = data.drop(dv_name, axis=1)
    if stratify:
        stratify = target
    else:
        stratify = None
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        train_size=train_size,
        test_size=(1 - train_size),
        random_state=random_state,
        stratify=stratify,
    )
    return x_train, x_test, y_train, y_test
