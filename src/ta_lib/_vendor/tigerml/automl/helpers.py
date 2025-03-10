import pandas as pd
from tigerml.core.scoring import SCORING_OPTIONS


def predict_values(pipeline_obj, x_data, proba=True):
    results = []
    results_proba = []
    error_predict = False
    error_predict_proba = False
    try:
        results = pipeline_obj.predict(x_data)
    except Exception:
        try:
            results = pipeline_obj.predict(x_data.values)
        except Exception:
            error_predict = True
    if proba:
        try:
            results_proba = pipeline_obj.predict_proba(x_data)
        except Exception:
            try:
                results_proba = pipeline_obj.predict_proba(x_data.values)
            except Exception:
                error_predict_proba = True

        return (error_predict, error_predict_proba, results, results_proba)
    else:
        return (error_predict, results)


def fit_and_predict(pipeline_obj, X_train, y_train, X_test, proba=True):
    error = [
        "fit",
        "predict_train",
        "predict_proba_train",
        "predict_test",
        "predict_proba_test",
    ]
    error = dict(zip(error, [False] * len(error)))
    results_train = []
    results_proba_train = []
    results_test = []
    results_proba_test = []
    try:
        pipeline_obj.fit(X_train, y_train)
    except Exception:
        try:
            pipeline_obj.fit(X_train.values, y_train)
        except Exception:
            error = error.fromkeys(error, True)
    if error["fit"] is False:
        if proba:
            (
                error["predict_train"],
                error["predict_proba_train"],
                results_train,
                results_proba_train,
            ) = predict_values(pipeline_obj, X_train)
            (
                error["predict_test"],
                error["predict_proba_test"],
                results_test,
                results_proba_test,
            ) = predict_values(pipeline_obj, X_test)
        else:
            error["predict_train"], results_train = predict_values(
                pipeline_obj, X_train, proba=False
            )
            error["predict_test"], results_test = predict_values(
                pipeline_obj, X_test, proba=False
            )

    if proba:
        return (
            error,
            results_train,
            results_proba_train,
            results_test,
            results_proba_test,
        )
    else:
        return (error, results_train, results_test)


def num_of_negs(y_data, neg_val=0):
    return len(y_data[y_data == neg_val])


def num_of_pos(y_data, pos_val=1):
    return len(y_data[y_data == pos_val])


def fit_predict_pipeline(pipeline, x_train, y_train, x_test=None, prob=False, n_jobs=1):
    from tpot.builtins.stacking_estimator import StackingEstimator

    for step in pipeline:
        if isinstance(step, StackingEstimator):
            step.estimator.n_jobs = n_jobs
        else:
            step.n_jobs = n_jobs
    pipeline.fit(x_train, y_train)
    if prob and hasattr(pipeline, "predict_proba"):
        yhat_is_prob = True
        yhat_train = pipeline.predict_proba(x_train)[:, 1]
        if x_test is not None:
            yhat_test = pipeline.predict_proba(x_test)[:, 1]
        else:
            yhat_test = None
    else:
        yhat_is_prob = False
        yhat_train = pipeline.predict(x_train)
        if x_test is not None:
            yhat_test = pipeline.predict(x_test)
        else:
            yhat_test = None
    return yhat_train, yhat_test, yhat_is_prob


def get_metrics_score(
    task,
    y_train,
    yhat_train,
    y_test=None,
    yhat_test=None,
    yhat_is_prob=True,
    multi_class=False,
):

    if task == "classification":
        metrics_dict = SCORING_OPTIONS.classification.copy()
        if multi_class:
            remove_keys = {
                "f1_score",
                "precision",
                "recall",
                "roc_auc",
                "balanced_accuracy",
            }
            for key in remove_keys.intersection(set(metrics_dict.keys())):
                metrics_dict.pop(key)
        if yhat_is_prob:
            yhat_train_prob = yhat_train.copy()
            yhat_train = (yhat_train_prob > 0.5).astype(int)
            if yhat_test is not None:
                yhat_test_prob = yhat_test.copy()
                yhat_test = (yhat_test_prob > 0.5).astype(int)
        else:
            yhat_train_prob = None
            yhat_test_prob = None
    elif task == "regression":
        metrics_dict = SCORING_OPTIONS.regression.copy()
        metrics_dict.pop("Explained Variance")
    else:
        raise Exception("Incorrect task input")
    scores_dict = {}
    for metric, metric_details in metrics_dict.items():
        func = metric_details["func"]
        default_params = {}
        if "default_params" in metric_details:
            default_params = metric_details["default_params"]
        scores_dict[metric] = {}
        if yhat_train is not None:
            params = [y_train]
            if metric in ["log_loss", "roc_auc"]:
                params.append(yhat_train_prob)
            else:
                params.append(yhat_train)
            if params[1] is not None:
                scores_dict[metric]["train"] = round(func(*params, **default_params), 4)
            else:
                scores_dict[metric]["train"] = "NA"
        if yhat_test is not None:
            params = [y_test]
            if metric in ["log_loss", "roc_auc"]:
                params.append(yhat_test_prob)
            else:
                params.append(yhat_test)
            if params[1] is not None:
                scores_dict[metric]["test"] = round(func(*params, **default_params), 4)
            else:
                scores_dict[metric]["test"] = "NA"
    dict_of_df = {k: pd.DataFrame([v]) for k, v in scores_dict.items()}
    scores_df = pd.concat(dict_of_df, axis=1)
    scores_df.columns.set_names(["metric", "dataset"], inplace=True)
    return scores_df
