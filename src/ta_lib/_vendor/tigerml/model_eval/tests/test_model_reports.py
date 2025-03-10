import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tigerml.core.reports import create_report
from tigerml.model_eval import ClassificationReport, RegressionReport
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


def regression_data():
    """Get regression data."""
    california = fetch_california_housing()
    X = pd.DataFrame(california["data"], columns=california["feature_names"])
    y = california["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_data():
    """Get classification data."""
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
    y = cancer["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def multi_classification_data():
    """Get classification data."""
    iris = load_iris()
    X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train, X_test, y_train, y_test


def _get_report_object(
    is_classification,
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    report_option=1,
    include_test=False,
):
    kwargs = {"y_train": y_train}
    if report_option == 1:
        kwargs.update({"model": model, "refit": True, "x_train": x_train})
        if include_test:
            kwargs.update({"x_test": x_test, "y_test": y_test})
    elif report_option == 2:
        model.fit(x_train, np.ravel(y_train))
        if is_classification:
            yhat_train = model.predict_proba(x_train)
            yhat_test = model.predict_proba(x_test)
        else:
            yhat_train = model.predict(x_train)
            yhat_test = model.predict(x_test)
        kwargs.update({"refit": False, "yhat_train": yhat_train})
        if include_test:
            kwargs.update({"y_test": y_test, "yhat_test": yhat_test})
    if is_classification:
        report_obj = ClassificationReport(**kwargs)
    else:
        report_obj = RegressionReport(**kwargs)
    return report_obj


DATASETS = {
    "regression": regression_data(),
    "classification": classification_data(),
    "multi-classification": multi_classification_data(),
}

ERRORBUCKETS_SPEC = {
    "type": "perc",
    "edges": [-0.2, 0.2],
    "labels": [
        "under predictions (-inf,-0.2]",
        "correct predictions",
        "over predictions [0.2, inf)",
    ],
    "top_n_cols": 15,
}

MODELS = {
    "regression": {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor(),
    },
    "classification": {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "XGBClassifier": XGBClassifier(),
    },
    "multi-classification": {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "XGBClassifier": XGBClassifier(),
    },
}

MODELS_PIPELINE = {
    "regression": {
        "LinearRegressionPipeline": Pipeline(
            [("scaler", StandardScaler()), ("LinearRegression", LinearRegression())]
        ),
        "RandomForestRegressorPipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("RandomForestRegressor", RandomForestRegressor()),
            ]
        ),
        "XGBRegressorPipeline": Pipeline(
            [("scaler", StandardScaler()), ("XGBRegressor", XGBRegressor())]
        ),
    },
    "classification": {
        "LogisticRegressionPipeline": Pipeline(
            [("scaler", StandardScaler()), ("LogisticRegression", LogisticRegression())]
        ),
        "RandomForestClassifierPipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("RandomForestClassifier", RandomForestClassifier()),
            ]
        ),
        "XGBClassifierPipeline": Pipeline(
            [("scaler", StandardScaler()), ("XGBClassifier", XGBClassifier())]
        ),
    },
}


REPORT_OPTION = [1, 2]
INCLUDE_TEST = [False, True]
INCLUDE_SHAP = [False, True]


def run_performance_report(report_obj, name):
    name = "Testing/Perf_report--" + name
    report_dict = report_obj.get_performance_report()
    create_report(report_dict, name=name)


def run_interpretation_report(report_obj, errorbuckets_spec, include_shap, name):
    name = "Testing/Interp_report--" + name + "--include_shap-{}".format(include_shap)
    report_dict = report_obj.get_interpretation_report(
        include_shap=include_shap, errorbuckets_spec=errorbuckets_spec
    )
    create_report(report_dict, name=name)


def run_get_report(report_obj, errorbuckets_spec, include_shap, name):
    name = "Testing/Full_report--" + name + "--include_shap-{}".format(include_shap)
    report_obj.get_report(
        file_path=name, include_shap=include_shap, errorbuckets_spec=errorbuckets_spec
    )


def run_report(
    is_classification,
    multi_class=False,
    report_options=REPORT_OPTION,
    include_test_options=INCLUDE_TEST,
    include_shap_options=INCLUDE_SHAP,
    error_bucket_specs=ERRORBUCKETS_SPEC,
    perf_report=True,
    interp_report=True,
    full_report=False,
):
    if is_classification:
        if multi_class:
            models_dict = MODELS["multi-classification"]
            x_train, x_test, y_train, y_test = DATASETS["multi-classification"]
        else:
            models_dict = MODELS["classification"]
            x_train, x_test, y_train, y_test = DATASETS["classification"]
    else:
        models_dict = MODELS["regression"]
        x_train, x_test, y_train, y_test = DATASETS["regression"]
    for model_name, model in models_dict.items():
        print("==============================================")
        print("Model: " + model_name)
        for report_option in report_options:
            for include_test in include_test_options:
                print("----------------------------------------------")
                name = model_name + "--report_option-{}--include_test-{}".format(
                    report_option, include_test
                )
                obj = _get_report_object(
                    is_classification,
                    model,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    report_option=report_option,
                    include_test=include_test,
                )
                if full_report:
                    for include_shap in include_shap_options:
                        print(
                            "report_option={} & include_test={} & include_shap={}".format(
                                report_option, include_test, include_shap
                            )
                        )
                        run_get_report(
                            obj,
                            error_bucket_specs,
                            include_shap=include_shap,
                            name=name,
                        )
                else:
                    if perf_report:
                        print(
                            "report_option={} & include_test={}".format(
                                report_option, include_test
                            )
                        )
                        run_performance_report(obj, name)
                    if interp_report:
                        for include_shap in include_shap_options:
                            print(
                                "report_option={} & include_test={} & include_shap={}".format(
                                    report_option, include_test, include_shap
                                )
                            )
                            run_interpretation_report(
                                obj,
                                error_bucket_specs,
                                include_shap=include_shap,
                                name=name,
                            )


# Regression Report
# performance report only on all combinations
run_report(False, perf_report=True, interp_report=False, full_report=False)
# interpretation report on all combinations
run_report(False, perf_report=False, interp_report=True, full_report=False)
# full report for only report option-2, with including test data and including shap
run_report(
    False,
    report_options=[1, 2],
    include_test_options=[True, False],
    include_shap_options=[True],
    perf_report=False,
    interp_report=False,
    full_report=True,
)

# Classification Report
# performance report only on all combinations
run_report(True, perf_report=True, interp_report=False, full_report=False)
# interpretation report on all combinations
run_report(True, perf_report=False, interp_report=True, full_report=False)
# full report for only report option-2, with including test data and including shap
run_report(
    True,
    report_options=[1, 2],
    include_test_options=[True, False],
    include_shap_options=[True],
    perf_report=False,
    interp_report=False,
    full_report=True,
)

# Multi-Classification Report
# performance report only on all combinations
run_report(
    True, multi_class=True, perf_report=True, interp_report=False, full_report=False
)
# interpretation report on all combinations
run_report(
    True, multi_class=True, perf_report=False, interp_report=True, full_report=False
)
# full report for only report option-2, with including test data and including shap
run_report(
    True,
    multi_class=True,
    report_options=[1, 2],
    include_test_options=[True, False],
    include_shap_options=[True],
    perf_report=False,
    interp_report=False,
    full_report=True,
)
