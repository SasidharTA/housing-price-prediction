# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""
import logging
import numpy as np
import pandas as pd
import random
import tigerml.core.dataframe as td
from datetime import datetime
from tigerml.core.preprocessing import prep_data
from tigerml.core.utils import DictObject

from .backends.tpot import CONFIGS, STEPS

_LOGGER = logging.getLogger(__name__)

TASKS = DictObject({"classification": "classification", "regression": "regression"})


DATA_TYPES = DictObject(
    {
        "structured": "structured",
        "image": "image",
        "text": "text",
        "time_series": "time_series",
    }
)


def create_excel_report(report_dict, file_path):
    from tigerml.core.reports.excel.lib import process_report_content

    flatt_dict = process_report_content(report_dict)
    with pd.ExcelWriter(file_path) as writer:
        for key, table in flatt_dict.items():
            if key == "index_sheet" and isinstance(table, dict):
                index_df = table[""]
                index_df.to_excel(writer, sheet_name=key)
            else:
                table[0].data.to_excel(writer, sheet_name=key)
        worksheet = writer.sheets["index_sheet"]
        for row_num, value in enumerate(index_df["Sheet_No"].values):
            worksheet.write_url(
                row_num + 1,
                1,
                f"internal:'{value}'!A1",
                string=value,
            )


class AutoML:
    """AutoML toolkit for classification and regression models.

    Parameters
    ----------
    name: str, default=None
        Name of the Pipeline
    x_train: pandas.DataFrame, default=None
        IDV's train data
    y_train: pandas.Series/pandas.DataFrame, default=None
        Target train data
    x_test: pandas.DataFrame, default=None
        IDV's test data
    y_test: pandas.Series/pandas.DataFrame, default=None
        Target test data
    task: str, default= 'classification'
        accepted values: 'classification' / 'regression'
    data_type: str, default='structured'
        Data type, e.g. 'structured', 'image', 'text' or 'time_series'
    generations: int, default=100
        Number of generations for tpot genetic grid search
    population_size: int, default=100
        Population per generation for tpot genetic grid search.
        Total no.of.pipelines to search = (generations+1) * population_size
    search_space: dict, default=None
        Grid search space
    scoring_func: str, default=None
        Function to score the data
    cv_folds: int, default= 5
        Number of cross validation cuts
    preprocessing: dict, default=None
        Preprocessing steps
    selection: dict, default=None
        Selection criteria
    modeling: dict, default=None
        Type of algorithm
    max_time_mins: int, default=None
        Maximum run time in mins
    **kwargs: key-words arguments
        other applicable key-words arguments for TPOT configuration

    Examples
    --------
    >>> import pandas as pd
    >>> from tigerml.automl import AutoML, TASKS
    >>> c_data = pd.read_csv('Titanic_Raw.csv')
    >>> classifier = AutoML(generations=30, population_size=50,
        name='titanic',task=TASKS.classification)
    >>> classifier.prep_data(data=c_data.drop(['PassengerId', 'Name',
        'Ticket'], axis=1), dv_name='Survived', train_size=0.75,
        remove_na=False)
    >>> classifier.fit()
    >>> classifier.get_report(no_of_pipelines=10, format='.xlsx')
    """

    tasks = TASKS
    """dict: A dictionary for defining tasks type i.e. 'classification' or
     'regression'."""
    configs = DictObject({"structured": CONFIGS})
    """dict: A dictionary having configuration details for
    'structured' data type."""

    def __init__(
        self,
        name="",
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        task=TASKS.classification,
        data_type=DATA_TYPES.structured,
        search_space=None,
        scoring_func=None,
        cv_folds=None,
        preprocessing=None,
        selection=None,
        modeling=None,
        max_time_mins=None,
        **kwargs,
    ):
        """Automl class initialization."""
        _LOGGER.info("Initializing AutoML!!!!!!!")
        if "random_state" in kwargs:
            self.random_state = kwargs.pop("random_state")
        else:
            self.random_state = random.randint(1, 100)
        self.name = "{}{}{}_{}".format(name, "_" if name else "", data_type, task)
        self.task = task
        self.multi_class = False
        self.imbalanced = False
        self.start_time = None
        self.end_time = None
        self.data_type = data_type
        self.optimiser = None
        self.config = self._init_config(
            search_space, scoring_func, cv_folds, self.random_state, **kwargs
        )
        self.init_data(x_train, y_train, x_test, y_test)
        self._init_pipeline(**kwargs)
        if preprocessing is not None or selection is not None or modeling is not None:
            self.edit_config(
                modeling,
                preprocessing,
                selection,
                max_time_mins=max_time_mins,
                **kwargs,
            )

    def _init_config(
        self, search_space, scoring_func, cv_folds, random_state, **kwargs
    ):
        """Bring in config to the pipeline.

        Parameters
        ----------
        search_space: Grid search space
        scoring_func: Scoring function for test data
        cv_folds: Number of cuts for cross validation
        random_state: Random state seed
        **kwargs:

        Returns
        -------
        config: dict config object
        """
        if self.data_type == DATA_TYPES.structured:
            from .backends.tpot import CONFIGS

            if search_space == CONFIGS.imbalanced:
                self.imbalanced = True
            from .backends.tpot import get_tpot_config

            config = get_tpot_config(
                self.task,
                search_space,
                scoring_func,
                cv_folds,
                random_state=random_state,
                **kwargs,
            )
        return config

    def edit_config(self, modeling=None, preprocessing=None, selection=None, **kwargs):
        """Updating tpot config.

        Parameters
        ----------
        modeling: dict
            Type of algorithm
        preprocessing: dict
            Preprocessing steps
        selection: dict
            Selection criteria
        **kwargs: key-word arguments
            applicable key-word arguments for tpot config
        """
        if self.data_type == DATA_TYPES.structured:
            from .backends.tpot import edit_tpot_config

            self.config = edit_tpot_config(
                self,
                modeling=modeling,
                preprocessing=preprocessing,
                selection=selection,
                **kwargs,
            )

    def _init_pipeline(self, **kwargs):
        """Selection Algorithm.

        Parameters
        ----------
        **kwargs:

        """
        if self.task == self.tasks.classification:
            if self.data_type == DATA_TYPES.structured:
                from .backends.tpot import get_tpot_optimizer

                tpot_classifier = get_tpot_optimizer(task=self.tasks.classification)
                self.optimiser = tpot_classifier(
                    self.x_train, self.y_train, self.x_test, self.y_test, **self.config
                )
                self.optimiser._setup_config(self.config["config_dict"])
                self.config["config_dict"] = self.optimiser._config_dict
            elif self.data_type == DATA_TYPES.image:
                from .backends.autokeras import ImageClassifier

                self.optimiser = ImageClassifier(**kwargs)
            elif self.data_type == DATA_TYPES.text:
                from .backends.autokeras import TextClassifier

                self.optimiser = TextClassifier(**kwargs)
            else:
                raise Exception("Incorrect input")
        elif self.task == self.tasks.regression:
            if self.data_type == DATA_TYPES.structured:
                from .backends.tpot import get_tpot_optimizer

                tpot_regressor = get_tpot_optimizer(task=self.tasks.regression)
                self.optimiser = tpot_regressor(
                    self.x_train, self.y_train, self.x_test, self.y_test, **self.config
                )
                self.optimiser._setup_config(self.config["config_dict"])
                self.config["config_dict"] = self.optimiser._config_dict
            elif self.data_type == DATA_TYPES.image:
                from .backends.autokeras import ImageRegressor

                self.optimiser = ImageRegressor(**kwargs)
            elif self.data_type == DATA_TYPES.text:
                from .backends.autokeras import TextRegressor

                self.optimiser = TextRegressor(**kwargs)
            elif self.data_type == DATA_TYPES.time_series:
                from .backends.autots import Forecaster

                self.optimiser = Forecaster(**kwargs)
            else:
                raise Exception("Incorrect input")
        else:
            raise Exception("Incorrect input")

    def _clean_data(self):
        """Cleans self.data."""
        impute_num_na = "median"
        impute_cat_na = "mode"
        train_levels = 0
        test_levels = 0
        if self.x_train is not None and self.y_train is not None:
            if self.x_train.isnull().sum().sum() > 0:
                from tigerml.core.preprocessing import Imputer

                imputer = Imputer(
                    num_impute_method=impute_num_na, cat_impute_method=impute_cat_na
                )
                df = imputer.fit_transform(self.x_train)
                df = pd.DataFrame(df, columns=imputer.get_feature_names())
                df = df.infer_objects()
                self.x_train = td.DataFrame(df)
            if self.y_train.isnull().sum().sum() > 0:
                raise Exception("y cannot be missing in train")
            if self.data_type == self.tasks.classification:
                self.x_train = self.x_train.astype(np.float64)
                self.y_train = self.y_train.astype(np.float64)
                train_levels = len(set(self.y_train.values.ravel()))
        if self.x_test is not None and self.y_test is not None:
            if self.x_test.isnull().sum().sum() > 0:
                from tigerml.core.preprocessing import Imputer

                df = imputer.transform(self.x_test)
                df = pd.DataFrame(df, columns=imputer.get_feature_names())
                df = df.infer_objects()
                self.x_test = td.DataFrame(df)
            if self.y_test.isnull().sum().sum() > 0:
                raise Exception("y cannot be missing in test")
            if self.data_type == self.tasks.classification:
                self.x_test = self.x_test.astype(np.float64)
                self.y_test = self.y_test.astype(np.float64)
                test_levels = len(set(self.y_test.values.ravel()))
        if max(train_levels, test_levels) > 2:
            self.multi_class = True
        if self.optimiser is not None:
            self.optimiser.init_data(
                self.x_train, self.y_train, self.x_test, self.y_test, self.multi_class
            )

    def init_data(self, x_train, y_train, x_test, y_test):
        """Initiate train/test data validation and cleaning.

        Parameters
        ----------
        x_train: pandas.DataFrame, default=None
            IDV's train data
        y_train: pandas.Series/pandas.DataFrame, default=None
            Target train data
        x_test: pandas.DataFrame, default=None
            IDV's test data
        y_test: pandas.Series/pandas.DataFrame, default=None
            Target test data
        """
        if x_train is not None:
            assert (
                x_test is not None
            ), "Test data should be passed to evaluate the pipelines"
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self._clean_data()
        if self.imbalanced and self.y_train is not None:
            from sklearn.utils.class_weight import compute_sample_weight

            sample_weights = compute_sample_weight(
                class_weight="balanced", y=pd.concat([self.y_train, self.y_test])
            )
            self.train_weights = sample_weights[: len(self.y_train)]

    def prep_data(
        self,
        data,
        dv_name="y",
        train_size=0.75,
        remove_na=True,
        impute_num_na="",
        impute_cat_na="",
    ):
        """Data cleaning and creation of train & test set based on.

           the option provided.

        Parameters
        ----------
        data: pandas.DataFrame
            Input data to be processed
        dv_name: str
            Column name of the Target variable
        train_size: float, default=0.75
            Size of the train data in terms of % of overall sample in
            range [0, 1]
        remove_na: bool, default=True
            Flag if null values to be removed
        impute_num_na: str, default='median'
            imputation method for numerical columns, one of from "mean",
            "median", "mode", "constant" and "regression"
        impute_cat_na: str, default='mode'
            imputation method for categorical columns, one of from "mean",
            "median", "mode", "constant" and "regression".
        """
        if isinstance(data, str):
            data_df = pd.read_csv(data, engine="python")
        elif isinstance(data, pd.DataFrame):
            data_df = data
        else:
            raise Exception("data should either be a dataframe or" "path to a csv file")
        if (
            not remove_na and self.data_type == DATA_TYPES.structured
        ):  # If TPOT is used with na values in data,
            if not impute_num_na:  # impute with median.
                # That is TPOT default config
                impute_num_na = "median"
            _LOGGER.info(
                "Will impute missing numeric values with {}".format(impute_num_na)
            )
            if not impute_cat_na:
                impute_cat_na = "mode"
            _LOGGER.info(
                "Will impute missing" "categorical values with {}".format(impute_cat_na)
            )
        x_train, x_test, y_train, y_test = prep_data(
            data_df,
            dv_name=dv_name,
            train_size=train_size,
            remove_na=remove_na,
            impute_num_na=impute_num_na,
            impute_cat_na=impute_cat_na,
            random_state=self.random_state,
        )
        self.init_data(x_train, y_train, x_test, y_test)

    @property
    def model_config(self):
        """Model configuration information for tpot Optimizer.

        Returns
        -------
        config_dict: dict
            configuration dictionary
        """
        return self.optimiser.config

    def set_model_config(self, model_dict):
        """Setup a configuration dictionary for tpot optimiser."""
        return self.optimiser.set_config(model_dict)

    @property
    def search_params(self):
        """Get parameters for an estimator.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self.optimiser.params

    def set_search_params(self, **params):
        """Setup the search space parameters for TPOT optimiser."""
        return self.optimiser.set_params(**params)

    def fit(self, *args, **kwargs):
        """Fit the optimizer."""
        if self.x_train is None or self.y_train is None:
            raise Exception(
                "Train data is not initiated. Use init_data() or prep_data()."
            )
        self.start_time = datetime.now()
        _LOGGER.info("Optimization Start Time: {}".format(self.start_time))
        try:
            if self.imbalanced:
                _LOGGER.info("Adding sample weights for imbalanced data")
                kwargs.update({"sample_weight": self.train_weights})
            self.optimiser.fit(*args, **kwargs)
            _LOGGER.info("Search Run Successful")
        except Exception as e:
            raise Exception("Search Run Not Successful {}".format(e))
        self.end_time = datetime.now()
        time_elapsed = self.end_time - self.start_time
        _LOGGER.info("Optimization End Time: {}".format(self.end_time))
        _LOGGER.info("Time Elapsed for Optimization: {}".format(time_elapsed))

    def score(self, x_data, y_data):
        """Return the test evaluation metrics.

        Returns the optimized pipelines score on the given testing data using
        the user-specified scoring function

        Parameters
        ----------
        x_data: array-like {n_samples, n_features}
            features data
        y_data: array-like {n_samples}
            target values

        Returns
        -------
        accuracy_score: float
            The estimated test set accuracy
        """
        return self.optimiser.score(x_data, y_data)

    def get_searched_space(self):
        """Return steps and algorithm information for evaluated pipelines.

        Returns a pandas dataframe with steps in pipeline as columns and
        algorithm used as rows.

        Returns
        -------
        tpot_obj_sort : pandas.dataframe
            dataframe with evaluated pipelines information
        """
        return self.optimiser.get_searched_space()

    def get_search_profile(self):
        """Return number of pipelines evaluated for a step and algorithm.

        Returns
        -------
        search_profile: dict
            dictionary with evaluated pipeline information
        """
        return self.optimiser.get_search_profile()

    def get_run_metrics(self):
        """Return pipelines execution summary.

        Returns a dataframe which gives info about time taken for pipeline
        and no of pipelines evaluated.

        Returns
        -------
        output : pandas.dataframe
            dataframe with pipelines execution summary stats.
        """
        config_df = (
            pd.DataFrame([self.config]).transpose().rename(columns={0: "Details"})
        )
        return pd.concat([self.optimiser.get_run_metrics(), config_df], axis=0)

    def get_pipeline_metrics(self, no_of_pipelines=None):
        """Return a dataframe with top N pipelines and their evaluated metrics.

        Parameters
        ----------
        no_of_pipelines: int
            number of pipelines for which to get metrics evaluation

        Returns
        -------
        pipeline_metrics : pandas.dataframe
            dataframe with relevant metrices for pipelines
        confusion_metrics : pandas.dataframe
            confusion matrix for classification problems
        """
        self.set_benchmarks()
        return self.optimiser.get_pipeline_metrics(
            no_of_pipelines=no_of_pipelines, benchmark=self.benchmark
        )

    def get_trained_pipeline(self, pipeline_str):
        """Export a fitted pipeline.

        Parameters
        ----------
        pipeline_str: str
            pipeline name

        Returns
        -------
        pipeline: object
            fitted pipeline object
        """
        return self.optimiser.get_trained_pipeline(pipeline_str)

    def get_pipeline_code(self, pipeline_str):
        """Generate and export source code for a TPOT Pipeline.

        Parameters
        ----------
        pipeline_str: str
            pipeline name

        Returns
        -------
        pipeline_code: str
            The source code representing the pipeline
        """
        return self.optimiser.get_pipeline_code(pipeline_str)

    def set_benchmarks(self, type="linear"):
        """Setting up a benchmark model for comparison with other pipelines.

        Based on task type, define and fit the benchmark model
        on the train set

        Parameters
        ----------
        type: str, default='linear'
            Type of model for classification case i.e. 'linear'
            or anything else
        """
        if self.task == TASKS.classification:
            if type == "linear":
                from sklearn.linear_model import LogisticRegression

                self.benchmark = LogisticRegression()
            else:
                from sklearn.tree import DecisionTreeClassifier

                self.benchmark = DecisionTreeClassifier()
        else:
            from sklearn.linear_model import LinearRegression

            self.benchmark = LinearRegression()
        self.benchmark.fit(self.x_train, np.ravel(self.y_train))

    def _get_report_dict(self, no_of_pipelines=10):
        if (
            len(self.optimiser.evaluated_individuals_.keys())
            == self.optimiser.population_size
            and self.optimiser.generations != 1
        ):
            raise Exception(
                "AutoML not run yet. Please call .fit(). "
                "If it was called before, it was not successful. "
                "The issue could be with data input. Please run again."
            )
        report_dict = {}
        report_dict["leaderboard"] = {}
        top_pipelines_str = "top_" + (str(no_of_pipelines) + "_pipelines")
        report_dict[top_pipelines_str] = {}
        report_dict["run_summary"] = {}

        from tigerml.core.reports import Table, preset_styles, table_styles

        search_stats = self.get_searched_space()
        search_stats_styled = Table(search_stats)
        search_stats_styled.apply_cell_format(
            {"width": table_styles.get_max_width}, cols=STEPS
        )
        search_stats_styled.apply_cell_format(
            {"width": table_styles.get_header_width},
            cols=[x for x in search_stats.columns if "pipeline" in x],
        )
        report_dict["leaderboard"].update({"pipelines": [search_stats_styled]})
        conf_matrices = None
        if self.task is self.tasks.classification:
            pipeline_metrics, conf_matrices = self.get_pipeline_metrics(
                no_of_pipelines=no_of_pipelines
            )
            metrics_formatted = Table(pipeline_metrics)
            good_metrics = [
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "roc_auc",
                "balanced_accuracy",
            ]
            metrics_formatted.apply_cell_format(
                table_styles.percentage_format,
                cols=[
                    x
                    for x in pipeline_metrics.columns
                    if any([col in x for col in good_metrics])
                ],
            )
            good_metrics += [
                x
                for x in pipeline_metrics.columns.get_level_values("metric")
                if "predicted" in x
                and x.split("_for_")[0].rsplit("_", maxsplit=1)[1]
                == x.split("_for_")[-1]
            ]
            metrics_formatted.apply_conditional_format(
                cols=[
                    x
                    for x in pipeline_metrics.columns
                    if any([col in x for col in good_metrics])
                ],
                options=preset_styles.more_is_good,
            )
            bad_metrics = ["log_loss"]
            bad_metrics += [
                x
                for x in pipeline_metrics.columns.get_level_values("metric")
                if "predicted" in x
                and x.split("_for_")[0].rsplit("_", maxsplit=1)[1]
                != x.split("_for_")[-1]
            ]
            metrics_formatted.apply_conditional_format(
                cols=[
                    x
                    for x in pipeline_metrics.columns
                    if any([col in x for col in bad_metrics])
                ],
                options=preset_styles.more_is_bad,
            )
        else:
            pipeline_metrics = self.get_pipeline_metrics(
                no_of_pipelines=no_of_pipelines
            )
            metrics_formatted = Table(pipeline_metrics)
            metrics_formatted.apply_cell_format(
                preset_styles.percent,
                cols=[
                    x
                    for x in pipeline_metrics.columns
                    if any(
                        [col in x for col in ["Explained Variance", "MAPE", "WMAPE"]]
                    )
                ],
            )
            metrics_formatted.apply_conditional_format(
                cols=[
                    x
                    for x in pipeline_metrics.columns
                    if any([col in x for col in ["Explained Variance", "R^2"]])
                ],
                options=preset_styles.more_is_good,
            )
            metrics_formatted.apply_conditional_format(
                cols=[
                    x
                    for x in pipeline_metrics.columns
                    if any([col in x for col in ["MAPE", "WMAPE", "MAE", "RMSE"]])
                ],
                options=preset_styles.more_is_bad,
            )
        metrics_formatted.apply_cell_format(
            {"width": table_styles.get_max_width},
            cols=list(pipeline_metrics.columns)[2:],
        )
        metrics_formatted.apply_cell_format(
            {"width": table_styles.get_max_width}, cols=STEPS
        )
        metrics_formatted.sort_columns(start=["pipeline_str"] + STEPS)
        report_dict[top_pipelines_str].update({"metrics": [metrics_formatted]})
        if conf_matrices is not None:
            conf_matrices = conf_matrices.reset_index()
            conf_matrices_formatted = Table(conf_matrices)
            conf_matrices_formatted.apply_cell_format(
                preset_styles.percent,
                cols=[x for x in conf_matrices.columns if "_normalized" in x[1]],
            )
            conf_matrices_formatted.apply_cell_format(
                {"width": table_styles.get_header_width},
                cols=list(conf_matrices.columns),
                index=[0, 1],
            )
            report_dict[top_pipelines_str].update(
                {"conf_matrices": [conf_matrices_formatted]}
            )
        run_stats = self.get_run_metrics()
        search_profile = self.get_search_profile()
        run_stats_formatted = Table(run_stats)
        run_stats_formatted.apply_cell_format(
            table_styles.time_format, rows=["Optimization Time Elapsed In Seconds"]
        )
        run_stats_formatted.apply_cell_format(
            {"bold": True},
            rows=["Optimization Time Elapsed In Seconds", "No of pipelines evaluated"],
        )
        run_stats_formatted.apply_cell_format(
            [{"width": table_styles.get_max_width}, table_styles.left_align], index=True
        )
        run_stats_formatted.apply_cell_format(
            {"width": 20}, cols=list(run_stats.columns)
        )
        report_dict["run_summary"].update({"run_info": [run_stats_formatted]})
        for index, key in enumerate(search_profile.keys()):
            step_df = pd.DataFrame([search_profile[key]]).transpose()
            step_df = step_df.rename(columns={0: "No of Pipelines"})
            step_df.index = step_df.index.rename(key)
            step_df_styled = Table(step_df)
            step_df_styled.apply_cell_format(
                {"width": table_styles.get_max_width}, index=True
            )
            step_df_styled.apply_cell_format(
                {"width": table_styles.get_header_width}, cols=list(step_df.columns)
            )
            report_dict["run_summary"].update(
                {"pipelines_with_{}".format(key): [step_df_styled]}
            )
        return report_dict, metrics_formatted

    def get_report(self, no_of_pipelines=10, format=".html", reports_path=""):
        """Generate and save the AutoML report.

        Parameters
        ----------
        no_of_pipelines: int, default=10
            number of pipelines for which metrics
            evaluation to be add in the report
        format: str
            report format, one of from ".html", ".pdf" and ".xlsx"
        reports_path: str
            valid directory path where reports to be saved
        """
        if (
            len(self.optimiser.evaluated_individuals_.keys())
            == self.optimiser.population_size
            and self.optimiser.generations != 1
        ):
            raise Exception(
                "AutoML not run yet. Please call .fit(). "
                "If it was called before, it was not successful. "
                "The issue could be with data input. Please run again."
            )
        # workbook = self._create_workbook()
        from tigerml.core.reports import create_report

        time = self.start_time if self.start_time else datetime.now()
        op_filename = (
            reports_path + self.name + "_at_" + str(time.strftime("%Y-%m-%d_%H-%M-%S"))
        )

        report_dict, metrics_formatted = self._get_report_dict(
            no_of_pipelines=no_of_pipelines
        )
        if format == ".html":
            create_report(report_dict, name=op_filename, format=format)
        elif format == ".xlsx":
            if op_filename[-5:] != ".xlsx":
                op_filename = op_filename + format
            create_excel_report(report_dict, op_filename)
        else:
            raise Exception("format should be '.html' or '.xlsx'")
        _LOGGER.info("AutoML report saved!")
        return metrics_formatted.styler

    def _restore_from_report(self, path_to_report):
        # TODO: Finish this
        pass
