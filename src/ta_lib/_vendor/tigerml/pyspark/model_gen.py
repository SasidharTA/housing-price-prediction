"""Module to carryout Model Generation and Training."""


# -----------------------------------------------------------------------
# Estimators and Models Currently Valid
# (Scope can be expanded to a lot more algorithms like NN etc.,)
# -----------------------------------------------------------------------
model_objects = {
    "regression": {
        "aft_survival": {
            "estimator": "pyspark.ml.regression.AFTSurvivalRegression",
            "model": "pyspark.ml.regression.AFTSurvivalRegressionModel",
        },
        "decision_tree": {
            "estimator": "pyspark.ml.regression.DecisionTreeRegressor",
            "model": "pyspark.ml.regression.DecisionTreeRegressionModel",
        },
        "gbt": {
            "estimator": "pyspark.ml.regression.GBTRegressor",
            "model": "pyspark.ml.regression.GBTRegressorModel",
        },
        "glm": {
            "estimator": "pyspark.ml.regression.GeneralizedLinearRegression",
            "model": "pyspark.ml.regression.GeneralizedLinearRegressionModel",
        },
        "isotonic_regression": {
            "estimator": "pyspark.ml.regression.IsotonicRegression",
            "model": "pyspark.ml.regression.IsotonicRegressionModel",
        },
        "linear_regression": {
            "estimator": "pyspark.ml.regression.LinearRegression",
            "model": "pyspark.ml.regression.LinearRegressionModel",
        },
        "rf": {
            "estimator": "pyspark.ml.regression.RandomForestRegressor",
            "model": "pyspark.ml.regression.RandomForestRegressorModel",
        },
    },
    "classification": {
        "logistic": {
            "estimator": "pyspark.ml.classification.LogisticRegression",
            "model": "pyspark.ml.classification.LogisticRegressionModel",
        },
        "decision_tree": {
            "estimator": "pyspark.ml.classification.DecisionTreeClassifier",
            "model": "pyspark.ml.classification.DecisionTreeClassifierModel",
        },
        "gbt": {
            "estimator": "pyspark.ml.classification.GBTClassifier",
            "model": "pyspark.ml.classification.GBTClassifierModel",
        },
        "rf": {
            "estimator": "pyspark.ml.classification.RandomForestClassifier",
            "model": "pyspark.ml.classification.RandomForestClassifierModel",
        },
        "naive_bayes": {
            "estimator": "pyspark.ml.classification.NaiveBayes",
            "model": "pyspark.ml.classification.NaiveBayesModel",
        },
        "multi_layer_perceptron": {
            "estimator": "pyspark.ml.classification.MultilayerPerceptronClassifier",
            "model": "pyspark.ml.classification.MultilayerPerceptronClassifierModel",
        },
        "one_vs_rest": {
            "estimator": "pyspark.ml.classification.OneVsRest",
            "model": "pyspark.ml.classification.OneVsRestModel",
        },
    },
    "clustering": {},
    "recommendation": {},
}

valid_model_types = [
    x for x in model_objects.keys() if len(model_objects[x].keys()) > 0
]
