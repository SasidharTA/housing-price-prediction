import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tigerml.model_eval import ClassificationComparison, RegressionComparison

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


def _get_report_object(
    is_classification, models, x_train, x_test, y_train, y_test, yhats_flag
):
    fitted_models = []
    yhats = {}
    for model in models:
        model.fit(x_train, np.ravel(y_train))
        fitted_models.append(model)
        if yhats_flag:
            if len(yhats) == 0:
                if hasattr(model, "predict_proba"):
                    yhats["linear"] = model.predict_proba(x_test)
                else:
                    yhats["linear"] = model.predict(x_test)
            else:
                if hasattr(model, "predict_proba"):
                    yhats["rf"] = model.predict_proba(x_test)
                else:
                    yhats["rf"] = model.predict(x_test)

    kwargs = {"models": fitted_models, "x": x_test, "y": y_test}
    if yhats_flag:
        kwargs = {"x": x_test, "y": y_test, "yhats": yhats}
    if is_classification:
        report_obj = ClassificationComparison(**kwargs)
    else:
        report_obj = RegressionComparison(**kwargs)
    return report_obj


DATASETS = {"regression": regression_data(), "classification": classification_data()}

MODELS = {
    "regression": {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
    },
    "classification": {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
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
    },
}

MODELS_AND_PIPELINE = {
    "regression": {
        "LinearRegressionPipeline": Pipeline(
            [("scaler", StandardScaler()), ("LinearRegression", LinearRegression())]
        ),
        "RandomForestRegressor": RandomForestRegressor(),
    },
    "classification": {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifierPipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("RandomForestClassifier", RandomForestClassifier()),
            ]
        ),
    },
}

YHATS = {
    "regression": {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
    },
    "classification": {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
    },
}

LOAD_TEST = {
    "regression": {
        "M1": LinearRegression(),
        "M2": LinearRegression(),
        "M3": RandomForestRegressor(n_estimators=100),
        "M4": RandomForestRegressor(n_estimators=200),
        "M5": RandomForestRegressor(n_estimators=300),
        "M6": RandomForestRegressor(n_estimators=400),
    },
    "classification": {
        "M1": LogisticRegression(),
        "M2": LogisticRegression(),
        "M3": RandomForestClassifier(n_estimators=100),
        "M4": RandomForestClassifier(n_estimators=200),
        "M5": RandomForestClassifier(n_estimators=300),
        "M6": RandomForestClassifier(n_estimators=400),
    },
}


def run_report(is_classification):
    test_list = [
        "MODELS",
        "MODELS_PIPELINE",
        "MODELS_AND_PIPELINE",
        "YHATS",
        "LOAD_TEST",
    ]
    for i, MODELS_ in enumerate(
        [MODELS, MODELS_PIPELINE, MODELS_AND_PIPELINE, YHATS, LOAD_TEST]
    ):
        if is_classification:
            models_list = list(MODELS_["classification"].values())
            x_train, x_test, y_train, y_test = DATASETS["classification"]
            name = "Testing/ComparisionReport--Classification--{}".format(test_list[i])
        else:
            models_list = list(MODELS_["regression"].values())
            x_train, x_test, y_train, y_test = DATASETS["regression"]
            name = "Testing/ComparisionReport--Regression--{}".format(test_list[i])
        yhats_flag = False
        if test_list[i] == "YHATS":
            yhats_flag = True
        report_obj = _get_report_object(
            is_classification,
            models_list,
            x_train,
            x_test,
            y_train,
            y_test,
            yhats_flag=yhats_flag,
        )
        report_obj.get_report(file_path=name)


# for RegressionComparision
run_report(is_classification=False)

# for ClassificationComparision
run_report(is_classification=True)
