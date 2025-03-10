"""TPOT Drive."""
import logging
import numpy as np
import pandas as pd
import random
import sys
import warnings
from datetime import datetime
from deap import creator
from sklearn.metrics import SCORERS, make_scorer
from tigerml.core.utils import DictObject, import_from_module_path
from tpot import TPOTClassifier, TPOTRegressor
from tpot.export_utils import export_pipeline

_LOGGER = logging.getLogger(__name__)

TASKS = DictObject({"classification": "classification", "regression": "regression"})


CUSTOM_SCORING_OPTIONS = DictObject(
    {
        "classification": {
            "log_loss": "log_loss",
            "f1_score": "f1_score",
            "balanced_accuracy": "balanced_accuracy",
        },
        "regression": {"MAPE": "MAPE", "WMAPE": "WMAPE"},
    }
)

STEPS_DICT = DictObject(
    {
        "preprocessing": [
            "preprocessing",
            "tpot.builtins",
            "kernel_approximation",
            "decomposition",
            "FeatureAgglomeration",
        ],
        "selection": ["feature_selection"],
    }
)

STEPS = list(STEPS_DICT.keys()) + ["modeling"]


default_params = {
    "generations": 100,
    "population_size": 100,
    "offspring_size": None,
    "mutation_rate": 0.9,
    "crossover_rate": 0.1,
    "subsample": 1.0,
    "n_jobs": 1,
    "max_time_mins": None,
    "max_eval_time_mins": 5,
    "random_state": random.randint(1, 100),
    "config_dict": None,
    "template": None,
    "warm_start": True,
    "memory": "auto",
    "use_dask": False,
    "early_stop": 5,
    "periodic_checkpoint_folder": "int_output",
    "verbosity": 2,
    "disable_update_check": False,
}

default = None
light = "TPOT light"
mdr = "TPOT MDR"
sparse = "TPOT sparse"
imbalanced = "imbalanced"
popular = "popular"
custom_configs = {"classification": [popular, imbalanced], "regression": [popular]}

CONFIGS = DictObject(
    {default: default, light: light, mdr: mdr, sparse: sparse, imbalanced: imbalanced}
)

if "linux" in sys.platform:
    default_params["n_jobs"] = -1

default_classification_scoring = "roc_auc"
default_regression_scoring = "r2"
default_classification_cv = 5
default_regression_cv = 5


def get_tpot_config(task, search_space, scoring_func, cv_folds, **kwargs):
    """Prepare config dictionary for the TPOTOptimizer class.

    Parameters
    ----------
        task : str
        search_spae : str
        scoring_func : scorer  or str
        cv_folds : int
    Returns
    -------
        params : dict
    """
    from sklearn.model_selection import StratifiedKFold

    params = default_params.copy()
    if "random_state" in kwargs:
        params["random_state"] = kwargs["random_state"]
    if search_space:
        if search_space in custom_configs[task]:
            search_space = import_from_module_path(
                "tigerml.automl.custom_configs.{}.{}".format(task, search_space)
            )
        params["config_dict"] = search_space
    if not scoring_func:
        if task == TASKS.classification:
            scoring_func = default_classification_scoring
        else:
            scoring_func = default_regression_scoring
    custom_scoring = False
    if not callable(scoring_func):
        custom_scorings = CUSTOM_SCORING_OPTIONS[task]
        if scoring_func not in SCORERS.keys() or scoring_func == "balanced_accuracy":
            custom_scoring = True
            if scoring_func in custom_scorings:
                from tigerml.core.scoring import SCORING_OPTIONS

                scoring_details = SCORING_OPTIONS[task][scoring_func]
                scoring_func = make_scorer(
                    scoring_details["func"],
                    greater_is_better=scoring_details["more_is_better"],
                )
            else:
                # makes a scorer from a performance
                raise Exception(
                    "{} scoring function does not exist. Pass the function that can compute the score".format(
                        scoring_func
                    )
                )
    else:
        custom_scoring = True
        module = getattr(scoring_func, "__module__", None)
        scoring_name = scoring_func.__name__
        if not module.startswith("sklearn.metrics.scorer") and not module.startswith(
            "sklearn.metrics.tests."
        ):
            greater_is_better = (
                "loss" not in scoring_name and "error" not in scoring_name
            )
            scoring_func = make_scorer(
                scoring_func, greater_is_better=greater_is_better
            )
            warnings.warn(
                "WARNING: The given scoring function is considered to be {} (Score is minimized if loss or error "
                "is in the function name). It is recommended to pass a sklearn scoreer object using "
                "make_scorer to avoid this assumption.".format(
                    "maximized" if greater_is_better else "minimized"
                )
            )
    if custom_scoring:
        params["n_jobs"] = 1
        warnings.warn(
            "WARNING: When you use a custom scoring function, multiple cores cannot be used. "
            "This might slowdown the overall process."
        )
    params["scoring"] = scoring_func
    if cv_folds:
        if task == TASKS.classification:
            params["cv"] = StratifiedKFold(
                n_splits=cv_folds, shuffle=False, random_state=params["random_state"]
            )
        else:
            params["cv"] = cv_folds
    else:
        if task == TASKS.classification:
            params["cv"] = default_classification_cv
        else:
            params["cv"] = default_regression_cv
    for key in [key for key in kwargs.keys() if key in params.keys()]:
        params[key] = kwargs[key]
    _LOGGER.info("Random state for the run is: {}".format(params["random_state"]))
    return params


def edit_tpot_config(
    automl, modeling=None, preprocessing=None, selection=None, **kwargs
):
    """Update TPOToptimizer config.

    Parameters
    ----------
        automl : dict
        modeling : dict, default None
        preprocessing : dict, default None
        selection : dict, default None
    Returns
    ------
        Params : updated config dict
    """
    params = automl.config
    if preprocessing is not None:
        if not isinstance(preprocessing, dict):
            raise Exception("Preprocessing should be a dict")
        for key in [
            x
            for x in params["config_dict"].keys()
            if [y for y in STEPS_DICT["preprocessing"] if y in x]
        ]:
            params["config_dict"].pop(key)
        params["config_dict"].update(preprocessing)
    if selection is not None:
        if not isinstance(selection, dict):
            raise Exception("Selection should be a dict")
        for key in [
            x
            for x in params["config_dict"].keys()
            if [y for y in STEPS_DICT["selection"] if y in x]
        ]:
            params["config_dict"].pop(key)
        params["config_dict"].update(selection)
    if modeling is not None:
        if not isinstance(modeling, dict):
            raise Exception("Modeling should be a dict")
        for key in [
            x
            for x in params["config_dict"].keys()
            if not any(
                [y in x for y in STEPS_DICT["preprocessing"] + STEPS_DICT["selection"]]
            )
        ]:
            params["config_dict"].pop(key)
        params["config_dict"].update(modeling)
    for key in [key for key in kwargs.keys() if key in params.keys()]:
        params[key] = kwargs[key]
    automl.optimiser.set_config(params["config_dict"])
    _LOGGER.info("Search space dict keys: {}".format(params["config_dict"].keys()))
    return params


def get_step_and_algo(algo):
    """Return step and algo_name.

    Parameters
    ----------
        algo : str
    Returns
    -------
        step : str
        algo_name  : str
    """
    algo_name = algo.split(".")[-1]
    step_of_algo = [
        step
        for step in STEPS_DICT
        if [module for module in STEPS_DICT[step] if module in algo]
    ]
    if step_of_algo:
        step = step_of_algo[0]
    else:
        step = "modeling"
    return (step, algo_name)


class NewScorer:
    """Newscorer class initializer."""

    def __init__(self, pipeline, func, sign):
        """Newscorer class initializer."""
        self.pipeline = pipeline
        self.scoring = func
        self.sign = sign

    def compute(self, y_true, y_pred):
        """Newscorer class initializer."""
        # import pdb
        # pdb.set_trace()
        cv_score = self.scoring(y_true, y_pred)
        return cv_score


def get_tpot_optimizer(task):
    """Return TPOTOptimizer class.

    Paramters
    ----------
        task : str
    Returns
    -------
        TPOTOptimizer class
    """
    if task == "classification":
        base = TPOTClassifier
    elif task == "regression":
        base = TPOTRegressor
    else:
        raise Exception("Incorrect input for task")

    class TPOTOptimizer(base):
        def __init__(self, x_train, y_train, x_test, y_test, **kwargs):
            """Initialize TPOTOptimizer class.

            Parameters
            ----------
                x_train : pandas dataframe/series
                y_train : pandas dataframe/series
                x_test : pandas dataframe/series
                y_test : pandas dataframe/series
            """
            super().__init__(**kwargs)
            self.start_time = None
            self.end_time = None
            self.tasks = TASKS
            self.init_data(x_train, y_train, x_test, y_test)
            self.sample_weights = None

        @staticmethod
        def _format_data(data):
            """Cast a pandas object to float type."""
            return data.astype(np.float64)

        @property
        def task(self):
            """Return task."""
            return (
                self.tasks.classification
                if self.classification
                else self.tasks.regression
            )

        def init_data(
            self, x_train, y_train, x_test=None, y_test=None, multi_class=False
        ):
            """Intializes data set.

            Parameters
            ----------
                multi_class: bool
                x_train : pandas dataframe/series
                y_train : pandas dataframe/series
                x_test : pandas dataframe/series
                y_test : pandas dataframe/series
            """
            self.x_train = x_train
            self.y_train = np.ravel(y_train)
            self.x_test = x_test
            self.y_test = np.ravel(y_test)
            if isinstance(self.y_train, pd.DataFrame):
                self.y_train = self.y_train.iloc[:, 0]
            if isinstance(self.y_test, pd.DataFrame):
                self.y_test = self.y_test.iloc[:, 0]
            self.multi_class = multi_class

        def fit(self, *args, **kwargs):
            """Run TPOT optimization."""
            if self.x_train is None or self.y_train is None:
                raise Exception("Train data is null.")
            self.start_time = datetime.now()
            super().fit(
                pd.DataFrame(self.x_train), np.ravel(self.y_train), *args, **kwargs
            )
            self.end_time = datetime.now()
            return self

        def score(self, x_data, y_data, *args):
            """Return the test evalation metrics.

            Returns the optimized pipeline's score on the given testing data using
            the user-specified scoring function

            Parameters
            ----------
                x_data : pandas series
                y_data : pandas series

            Returns
            -------
                optimized pipeline's score
            """
            x_data = self._format_data(x_data)
            y_true = self._format_data(y_data)
            return super().score(x_data, y_true)

        def get_searched_space(self):
            """Return TPOT evaluated pipelines.

            Returns a pandas dataframe with steps in pipeline as columns and
            algorithm used as rows.

            Returns
            -------
                tpot_obj_sort : pandas dataframe
            """
            if not hasattr(self, "evaluated_individuals_"):
                raise Exception("The TPOT search is not done yet. Call .fit() first.")
            tpot_obj_sort = pd.DataFrame()
            for key in self.evaluated_individuals_.keys():
                pipeline_dict = {}

                # pipeline_dict['pipeline_str'] = key

                pipeline_dict["pipeline_name"] = self.clean_pipeline_string(key)
                pipeline_dict["pipeline_score"] = self.evaluated_individuals_[key][
                    "internal_cv_score"
                ]
                pipeline_steps = self.get_steps_in_pipeline(key)
                for key in pipeline_steps:
                    pipeline_steps[key] = ", ".join(pipeline_steps[key])
                pipeline_dict.update(pipeline_steps)
                tpot_obj_sort = pd.concat(
                    [tpot_obj_sort, pd.DataFrame([pipeline_dict])], ignore_index=True
                )
            tpot_obj_sort.sort_values(["pipeline_score"], ascending=False, inplace=True)
            tpot_obj_sort.index = tpot_obj_sort.reset_index().index.rename("rank") + 1
            return tpot_obj_sort

        def get_run_metrics(self):
            """Return execution summary.

            Returns a dataframe which gives info about time taken for pipeline
            and no of piplelines evaluated

            Returns
            -------
                rub_data : pandas dataframe
            """
            if not hasattr(self, "evaluated_individuals_"):
                _LOGGER.info("The TPOT search is not done yet. Call .fit() first.")
                return
            run_data = pd.DataFrame(
                [
                    [
                        len(self.evaluated_individuals_.keys()),
                        self.start_time,
                        self.end_time,
                        (self.end_time - self.start_time).seconds / 86400,
                        self.task,
                    ]
                ],
                columns=[
                    "No of pipelines evaluated",
                    "Optimization Start Time",
                    "Optimization End Time",
                    "Optimization Time Elapsed In Seconds",
                    "Model Type",
                ],
            )

            run_data = run_data.transpose()
            run_data = run_data.rename(
                columns=dict(zip(list(run_data.columns), ["Details"]))
            )
            return run_data

        def get_search_profile(self):
            """Return evaluated pipelines.

            Returns a dictionary of no of pipelines evaluated for a step and algorithm.

            Returns
            -------
                search_profile: dict of dict
            """
            search_profile = {}
            for step in STEPS_DICT:
                search_profile[step] = {}
            search_profile["modeling"] = {}
            for algo in self.config.keys():
                (step, algo_name) = get_step_and_algo(algo)
                search_profile[step][algo_name] = len(
                    self.get_pipelines_with_algo(algo_name)
                )
            return search_profile

        def get_top_n_pipelines(self, n):
            """Return leaderboard.

            Parameters
            ----------
                n: int, no. of top pipelines
            Returns
            ------
                pipelines : list of dict
            """
            sorted_pipelines = list(
                sorted(
                    self.evaluated_individuals_.items(),
                    key=lambda kv: kv[1]["internal_cv_score"],
                    reverse=True,
                )
            )[:n]
            pipelines = []
            for pipeline in sorted_pipelines:
                pipeline_dict = pipeline[1]
                pipeline_steps = self.get_steps_in_pipeline(pipeline[0])
                for key in pipeline_steps:
                    pipeline_steps[key] = ", ".join(pipeline_steps[key])
                pipeline_dict.update(pipeline_steps)
                pipeline_dict["pipeline_str"] = pipeline[0]
                evaluated_pipeline = creator.Individual.from_string(
                    pipeline_dict["pipeline_str"], self._pset
                )

                # pipeline_dict['pipeline_code'] = export_pipeline(evaluated_pipeline, self.operators,
                #                                                  self._pset, self._imputed, pipeline_score=None)

                pipeline_dict["pipeline_obj"] = self._toolbox.compile(
                    expr=evaluated_pipeline
                )
                pipelines.append(pipeline_dict)
            return pipelines

        def get_pipeline_metrics(self, no_of_pipelines=None, benchmark=None):
            """Return a dataframe with top pipelines and metrics.

            Parameters
            ----------
                no_of_pipelines : int
                benchmark : estimator

            Returns
            -------
                pipeline_metrics : pandas dataframe
                confusion_metrics : pandas dataframe
            """
            from sklearn.pipeline import make_pipeline
            from tigerml.automl.helpers import (
                fit_predict_pipeline,
                get_metrics_score,
            )
            from tigerml.core.scoring import confusion_matrix_df

            predict_prob = False
            if self.task == self.tasks.classification:
                predict_prob = True
            pipeline_metrics = pd.DataFrame()
            confusion_matrices = pd.DataFrame()
            top_n_pipelines = self.get_top_n_pipelines(n=no_of_pipelines)
            if benchmark:
                benchmark_pipeline = {}
                benchmark_pipeline["pipeline_str"] = "Baseline Model ({})".format(
                    benchmark.__class__.__name__
                )
                benchmark_pipeline["pipeline_obj"] = make_pipeline(benchmark)
                for step in STEPS:
                    benchmark_pipeline[step] = ""
                benchmark_pipeline["modeling"] = benchmark.__class__.__name__
                top_n_pipelines.append(benchmark_pipeline)
            extra_col_names = ["pipeline_str"]
            for (ind, pipeline) in enumerate(top_n_pipelines):
                if ind + 1 > no_of_pipelines:
                    _LOGGER.info("Processing Benchmark Pipeline.")
                else:
                    _LOGGER.info("Processing Pipeline: {}".format((ind + 1)))
                try:
                    start_time = datetime.now()
                    yhat_train, yhat_test, yhat_is_prob = fit_predict_pipeline(
                        pipeline=pipeline["pipeline_obj"],
                        x_train=self.x_train,
                        y_train=self.y_train,
                        x_test=self.x_test,
                        prob=predict_prob,
                        n_jobs=self.n_jobs,
                    )
                    report_fit_score_time = datetime.now()
                    _LOGGER.info(
                        "Pipeline fit & score time: {}".format(
                            report_fit_score_time - start_time
                        )
                    )
                    metrics = get_metrics_score(
                        task=self.task,
                        y_train=self.y_train,
                        yhat_train=yhat_train,
                        y_test=self.y_test,
                        yhat_test=yhat_test,
                        yhat_is_prob=yhat_is_prob,
                        multi_class=self.multi_class,
                    )
                    if self.task == self.tasks.classification:
                        cm = confusion_matrix_df(
                            self.y_train, yhat_train, self.y_test, yhat_test
                        ).reset_index()
                        cm_norm = confusion_matrix_df(
                            self.y_train,
                            yhat_train,
                            self.y_test,
                            yhat_test,
                            normalized=True,
                        )
                        confusion_matrix = pd.concat([cm, cm_norm], axis=1)
                        confusion_matrix["pipeline_str"] = pipeline["pipeline_str"]
                        confusion_matrices = pd.concat(
                            [confusion_matrices, confusion_matrix]
                        )
                        cm_flat = confusion_matrix_df(
                            self.y_train,
                            yhat_train,
                            self.y_test,
                            yhat_test,
                            flattened=True,
                        )
                        metrics = pd.concat([metrics, cm_flat], axis=1)
                    metrics_dict = metrics.to_dict()
                    for col in extra_col_names + STEPS:
                        key = col
                        if len(metrics.columns.levels) > 1:
                            key = [col] + [""] * (len(metrics.columns.levels) - 1)
                            key = tuple(key)
                        metrics_dict[key] = pipeline[col]
                    metrics = pd.DataFrame(metrics_dict)
                    metrics.columns.set_names(["metric", "dataset"], inplace=True)
                    pipeline_metrics = pd.concat([pipeline_metrics, metrics])
                    pipeline_metrics.index = (
                        pipeline_metrics.reset_index().index.rename("rank") + 1
                    )
                except Exception as e:
                    _LOGGER.exception(
                        "Following exception occurred while processing this pipeline: {}".format(
                            e
                        )
                    )
            extra_cols = [
                col
                for col in pipeline_metrics.columns
                if any([x for x in extra_col_names if x in col])
            ]
            pipeline_metrics = pipeline_metrics[
                extra_cols
                + [x for x in pipeline_metrics.columns if x not in extra_cols]
            ]
            if self.task == self.tasks.classification:
                confusion_matrices.reset_index(drop=True, inplace=True)
                confusion_matrices = pd.merge(
                    pipeline_metrics[("pipeline_str", "")].reset_index(),
                    confusion_matrices,
                    on=[("pipeline_str", "")],
                )
                confusion_matrices = confusion_matrices.set_index(
                    [("rank", ""), ("pipeline_str", ""), ("true_label", "")]
                )
                return (pipeline_metrics, confusion_matrices)
            else:
                return pipeline_metrics

        @property
        def config(self):
            return self._config_dict

        def set_config(self, model_dict):
            return self._setup_config(model_dict)

        @property
        def params(self):
            return self.get_params()

        def create_from_tpot_object(self, tpot_obj):
            """Create a TPOT object.

            Parameters
            ----------
                tpot_obj:

            Returns
            -------
                self : TPOT object
            """
            for att in [a for a in dir(tpot_obj) if not a.startswith("__")]:
                setattr(self, att, getattr(tpot_obj, att))
            return self

        def get_trained_pipeline(self, pipeline_str):
            """Export fitted pipeline.

            Parameters
            ----------
                pipeline_str: str

            Returns
            -------
                pipeline : Fit object
            """
            evaluated_pipeline = creator.Individual.from_string(
                pipeline_str, self._pset
            )
            pipeline = self._toolbox.compile(expr=evaluated_pipeline)
            pipeline.fit(self.x_train, self.y_train)
            return pipeline

        def get_pipeline_code(self, pipeline_str):
            """Export a pipeline.

            Parameters
            ----------
                pipeline_str: str

            Returns
            -------
                pipeline_code : Code for the pipeline
            """
            evaluated_pipeline = creator.Individual.from_string(
                pipeline_str, self._pset
            )
            pipeline_code = export_pipeline(
                evaluated_pipeline,
                self.operators,
                self._pset,
                self._imputed,
                pipeline_score=None,
            )
            return pipeline_code

        def get_steps_in_pipeline(self, pipeline):
            """Return list of steps completed in a pipeline.

            Parameters
            ----------
                pipeline: str

            Return
            ------
                steps_dict : list
            """
            steps = []
            while "(" in pipeline:
                first = pipeline.split("(")[0]
                steps += [x for x in self.config.keys() if first in x]
                pipeline = pipeline[len(first) + 1 :]
            steps = sorted(steps, reverse=True)
            steps_dict = dict(
                zip(
                    list(STEPS_DICT.keys()) + ["modeling"],
                    [list() for x in range(0, len(STEPS_DICT.keys()) + 1)],
                )
            )

            for step in steps:
                (step, algo_name) = get_step_and_algo(step)
                steps_dict[step].append(algo_name)
            return steps_dict

        def get_pipelines_with_algo(self, algo):
            """Return pipeline list for the algo.

            Parameters
            ----------
                algo : str
            Returns
            ------
                pipelines_dict : list of pipelines
            """
            if "." in algo:
                algo = algo.split(".")[-1]
            pipelines_dict = {}
            for key in [x for x in self.evaluated_individuals_.keys() if algo in x]:
                pipelines_dict.update({key: self.evaluated_individuals_[key]})
            return pipelines_dict

        def get_overfit_check_scorer(self, scorer):
            score_func = scorer._score_func
            sign = scorer._sign
            scorer_obj = NewScorer(self, score_func, sign)
            return make_scorer(
                scorer_obj.compute, greater_is_better=True if sign > 0 else False
            )

    return TPOTOptimizer
