import holoviews
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import composite, integers, sampled_from
from pytest import fixture
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.validation import check_is_fitted
from tigerml.core.preprocessing import prep_data
from tigerml.model_eval.plotters.evaluation import ClassificationEvaluation

from ..comparison import ClassificationComparison


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate Classification dataset ----------------
@composite
def classification_data(draw):
    """Creates dataset of sizes upto 100k using hypothesis library and makes it into classfication data using sklearn.make_classfication."""
    n_samples_val = draw(integers(min_value=1000, max_value=100000))
    # n_samples_val = draw(integers(min_value=100, max_value=1000))
    n_features_val = draw(integers(min_value=7, max_value=50))
    n_informative_val = draw(integers(min_value=3, max_value=n_features_val - 2))
    hypercube_val = draw(sampled_from([True, False]))
    random_state_val = draw(integers(min_value=10, max_value=1000))
    array_data = make_classification(
        n_samples=n_samples_val,
        n_features=n_features_val,
        n_informative=n_informative_val,
        hypercube=hypercube_val,
        random_state=random_state_val,
    )
    x_data = array_data[0]
    y_data = array_data[1]
    df = pd.DataFrame(
        data=x_data[0:, 0:],
        index=[i for i in range(x_data.shape[0])],
        columns=["Col_" + str(i + 1) for i in range(x_data.shape[1])],
    )
    df["DV"] = y_data
    return df


# ---------------------------------------------------------------------------------
# ----------- setting up ClassificationEvaluation object for the tests ------------
@fixture
def sample_report():
    """Simplifies the task of fitting and scoring the models created using ClassificationEvaluation and uses them in all the test cases later."""

    def _get_data(df, scoring=True, return_test_df=False):
        x_train, x_test, y_train, y_test = prep_data(df, dv_name="DV")
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr = model.fit(x_train, np.ravel(y_train))
        yhat_test = lr.predict_proba(x_test)
        yhat_train = lr.predict_proba(x_train)
        print("x_train.shape", x_train.shape)
        report = ClassificationEvaluation(
            model, x_train, y_train, x_test, y_test, yhat_train, yhat_test
        )

        return_val = [report, model, x_train, y_train]
        if return_test_df:
            return_val += [x_test, y_test]
        return return_val

    return _get_data


# ---------------------------------------------------------------------------------
# ---------------------- confusion_matrix testing ---------------------------------
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
@given(test_df=classification_data())
def test_confusion_matrix(test_df, sample_report):
    """Creates an expected confusion matrix using scikit-learn and then compares with tigerml confusion matrix to check for the accuaracy."""
    report, model, x_train, y_train = sample_report(test_df)
    report.yhat_train = report.yhat_train[:, 1]
    report.yhat_test = report.yhat_test[:, 1]
    confusion_mat = report.confusion_matrix()
    confusion_mat = confusion_mat["train"]
    # confusion matrix given by tigerml ClassificationEvaluation
    confusion_mat = confusion_mat[["predicted_0", "predicted_1"]]
    # Calculating expected confusion_matrix
    model.fit(x_train, np.ravel(y_train))
    y_predict = model.predict(x_train)
    # expected confusion matrix from the model
    exp_confusion_mat = confusion_matrix(y_train, y_predict)
    exp_confusion_mat = pd.DataFrame(
        exp_confusion_mat, columns=["predicted_0", "predicted_1"]
    )
    assert (confusion_mat == exp_confusion_mat).sum().sum() == 4


# # -----------------------------------------------------------------------------
# # ---------------------- get_metrics testing ---------------------------------
# @settings(max_examples=10, deadline=None)
# @given(test_df=classification_data())
# def test_get_metrics(test_df, sample_report):
#     """Checks for the accuracy of all the metrics used in tigerml classification module by comparing them with the
#     metrics calculated using scikit-learn metircs.
#     """
#     report, model, x_train, y_train = sample_report(test_df, scoring=False)
#     get_metrics_result = report._get_metrics_for_decision_threshold(0.5).transpose()
#     get_metrics_result.reset_index(inplace=True)
#     get_metrics_result.sort_values(by=["metric"], inplace=True, ignore_index=True)
#     # Calculating expected metric values
#     model.fit(x_train, y_train)
#     exp_y_pred = model.predict(x_train)
#     exp_y_proba = model.predict_proba(x_train)
#     # calculating all classification metrics using scikit-learn metrics.
#     exp_values = [
#         accuracy_score(y_train, exp_y_pred),
#         balanced_accuracy_score(y_train, exp_y_pred),
#         f1_score(y_train, exp_y_pred, average="weighted"),
#         log_loss(y_train, exp_y_proba),
#         precision_score(y_train, exp_y_pred, average="weighted"),
#         recall_score(y_train, exp_y_pred, average="weighted"),
#         roc_auc_score(y_train, exp_y_proba[:, 1]),
#     ]
#     metric_names = [
#         "accuracy",
#         "balanced_accuracy",
#         "f1_score",
#         "log_loss",
#         "precision",
#         "recall",
#         "roc_auc",
#     ]
#     exp_df = pd.DataFrame({"metric": metric_names, "value": exp_values})
#     assert (get_metrics_result[0] == exp_df["value"]).sum() == 7


# ---------------------------------------------------------------------------------
# ------------------------------ fit testing -------------------------------------
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
@given(test_df=classification_data())
def test_fit(test_df, sample_report):
    """Checks if the models are fitting on the data or not."""
    report, model, x_train, y_train = sample_report(test_df, scoring=False)
    # check_is_fitted() raises a 'NotFittedError' error if the model is no fitted
    check_is_fitted(model)


# # ---------------------------------------------------------------------------------
# # ---------------------- confusion_matrix_heatmap testing ------------------------
# @settings(max_examples=10, deadline=None)
# @given(test_df=classification_data())
# def test_confusion_matrix_heatmap(test_df, sample_report):
#     """Checks for the accuracy of values in the confusion matrix and then the plot type of the heatmap."""
#     report, model, x_train, y_train, x_test, y_test = sample_report(
#         test_df, return_test_df=True
#     )
#     cm_heatmap_vals = report.confusion_matrix()
#     #cm_heatmap_vals = cm_heatmap.data["value"]
#     # Calculating expected confusion_matrix
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     exp_cm_vals = confusion_matrix(y_test, y_predict)
#     assert ((cm_heatmap_vals == exp_cm_vals).sum() == 4) and (
#         type(cm_heatmap) == holoviews.element.raster.HeatMap
#     )


# ---------------------------------------------------------------------------------
# ------------------------ precision_recall_curve testing ------------------------
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
@given(test_df=classification_data())
def test_precision_recall_curve(test_df, sample_report):
    """Checks for the plot type of precision-recall curve."""
    report, model, x_train, y_train = sample_report(test_df)
    pr_curve = report.precision_recall_curve()
    assert type(pr_curve) == str


# ---------------------------------------------------------------------------------
# ------------------------------- roc_auc testing --------------------------------
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=(
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ),
)
@given(test_df=classification_data())
def test_roc_curve(test_df, sample_report):
    """Checks for the plot type of roc_auc curve."""
    report, model, x_train, y_train = sample_report(test_df)
    roc_auc_curve = report.roc_curve()
    assert type(roc_auc_curve) == str


# ---------------------------------------------------------------------------------
# ------------------------- ClassificationComparisonMixin testing -----------------


# @settings(max_examples=10, deadline=None)
# @given(test_df=classification_data())
# def test_regression_comparison_mixin(test_df):
#    """Checks for the correct plot types expected in comparison of models and the data shape for performance report."""
#    x_train, x_test, y_train, y_test = prep_data(test_df, dv_name="DV")
#    models = [LogisticRegression(), RidgeClassifier()]
#    report = ClassificationComparison(y=y_train, models=models, x=x_train, refit=True)
#    #report.fit(x_train, y_train)
#    #report.score(x_test, y_test)
#    metrics = report.perf_metrics()
#    conf_mat = report.confusion_matrices()
#    roc_curves = report.roc_curves()
#    pr_curves = report.pr_curves()
#    dt_analyses = report.threshold_analysis()
#    perf_report = report.get_performance_report()

#    # test for perf_metrics
#    assert metrics.data.shape == (len(models), 7)

#    # test for confusion matrix
#    assert type(conf_mat) == holoviews.core.spaces.HoloMap

#    # test for roc_curves and pr_curves
#    assert type(roc_curves) == holoviews.core.overlay.Overlay
#    assert type(pr_curves) == holoviews.core.overlay.Overlay

#    # test for threshold analyses
#    assert type(dt_analyses) == dict

#    # test for performance report
#    assert sorted(list(perf_report.keys())) == [
#        "confusion_matrices",
#        "gains_charts",
#        "lift_charts",
#        "performance_metrics",
#        "precision_recall_curves",
#        "roc_curves",
#        "threshold_analysis",
#    ]
