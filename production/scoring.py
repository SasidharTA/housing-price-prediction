import os.path as op
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
from ta_lib.core.api import (get_dataframe,
                             get_feature_names_from_column_transformer,
                             get_package_path, hash_object, load_dataset,
                             load_pipeline, register_processor, save_dataset, DEFAULT_ARTIFACTS_PATH)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""

    # Define the datasets to load
    input_features_ds = "test/sales/features"
    input_target_ds = "test/sales/target"
    output_ds = "score/sales/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # Load test datasets
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # Load the feature pipeline and training pipelines
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))

    # Start an MLflow run
    with mlflow.start_run(run_name="Model Scoring") as run:
        # Transform the test dataset
        test_X = get_dataframe(
            features_transformer.transform(test_X),
            get_feature_names_from_column_transformer(features_transformer),
        )
        test_X = test_X[curated_columns]

        # Make a prediction
        test_X["yhat"] = model_pipeline.predict(test_X)

        # Evaluate the model
        mse = mean_squared_error(test_y, test_X["yhat"])
        r2 = r2_score(test_y, test_X["yhat"])

        # Log metrics to MLflow
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)

        # Log the predictions as an artifact
        mlflow.log_artifact("scoring_predictions.csv", test_X.to_csv(index=False))

        # Store the predictions for any further processing
        save_dataset(context, test_X, output_ds)

        # Optionally log the trained model as an artifact
        mlflow.sklearn.log_model(model_pipeline, "model")

        # You can also log the model and any other important artifacts as needed
        # mlflow.log_artifact("artifact_path")

