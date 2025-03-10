import logging
import os.path as op
import mlflow
import mlflow.sklearn  # Required to log scikit-learn models

from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model and log with MLflow."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/sales/features"
    input_target_ds = "train/sales/target"

    # Start MLflow experiment and log the run
    with mlflow.start_run(run_name="Train Regression Model") as run:
        # Set tags (optional)
        mlflow.set_tag("model", "SKLStatsmodelOLS")
        mlflow.set_tag("type", "regression")

        # Log parameters (optional)
        mlflow.log_param("sampling_fraction", params.get("sampling_fraction", None))

        # Load training datasets
        train_X = load_dataset(context, input_features_ds)
        train_y = load_dataset(context, input_target_ds)

        # Load pre-trained feature pipelines and other artifacts
        curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
        features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

        # Sample data if needed. Useful for debugging/profiling purposes.
        sample_frac = params.get("sampling_fraction", None)
        if sample_frac is not None:
            logger.warning(f"The data has been sampled by fraction: {sample_frac}")
            sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
        else:
            sample_X = train_X
        sample_y = train_y.loc[sample_X.index]

        # Transform the training data
        train_X = get_dataframe(
            features_transformer.fit_transform(train_X, train_y),
            get_feature_names_from_column_transformer(features_transformer),
        )
        train_X = train_X[curated_columns]

        # Create training pipeline
        reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

        # Fit the training pipeline
        reg_ppln_ols.fit(train_X, train_y.values.ravel())

        # Log model to MLflow
        mlflow.sklearn.log_model(reg_ppln_ols, "model")

        # Save fitted training pipeline locally
        save_pipeline(
            reg_ppln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
        )



