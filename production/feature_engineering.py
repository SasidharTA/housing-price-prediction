import logging
import os.path as op
import mlflow
import mlflow.sklearn

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)

@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/sales/features"
    input_target_ds = "train/sales/target"
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # Start MLflow run
    with mlflow.start_run(run_name="Feature Engineering: Transform Features") as run:
        # Load datasets
        train_X = load_dataset(context, input_features_ds)
        train_y = load_dataset(context, input_target_ds)

        # Log initial dataset size
        mlflow.log_param("initial_row_count", len(train_X))
        mlflow.log_param("initial_column_count", len(train_X.columns))

        cat_columns = train_X.select_dtypes("object").columns
        num_columns = train_X.select_dtypes("number").columns

        # Treating Outliers
        outlier_transformer = Outlier(method=params["outliers"]["method"])
        train_X = outlier_transformer.fit_transform(train_X, drop=params["outliers"]["drop"])

        # Log outlier treatment details
        mlflow.log_param("outlier_method", params["outliers"]["method"])
        mlflow.log_param("drop_outliers", params["outliers"]["drop"])

        # NOTE: Using Pipeline to combine target encoding and imputation steps
        tgt_enc_simple_impt = Pipeline(
            [
                ("target_encoding", TargetEncoder(return_df=False)),
                ("simple_impute", SimpleImputer(strategy="most_frequent")),
            ]
        )

        # ColumnTransformer for applying different transformations to different columns
        features_transformer = ColumnTransformer(
            [
                # Categorical columns
                (
                    "tgt_enc",
                    TargetEncoder(return_df=False),
                    list(
                        set(cat_columns)
                        - set(["technology", "functional_status", "platforms"])
                    ),
                ),
                (
                    "tgt_enc_sim_impt",
                    tgt_enc_simple_impt,
                    ["technology", "functional_status", "platforms"],
                ),
                # Numeric columns
                ("med_enc", SimpleImputer(strategy="median"), num_columns),
            ]
        )

        # Check if the data should be sampled
        sample_frac = params.get("sampling_fraction", None)
        if sample_frac is not None:
            logger.warning(f"The data has been sampled by fraction: {sample_frac}")
            sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
        else:
            sample_X = train_X
        sample_y = train_y.loc[sample_X.index]

        # Train the feature engineering pipeline on the data
        train_X = get_dataframe(
            features_transformer.fit_transform(train_X, train_y),
            get_feature_names_from_column_transformer(features_transformer),
        )

        # Log the shape of the transformed dataset
        mlflow.log_param("transformed_row_count", len(train_X))
        mlflow.log_param("transformed_column_count", len(train_X.columns))

        # Curate columns by removing irrelevant ones
        curated_columns = list(
            set(train_X.columns.to_list())
            - set(
                [
                    "manufacturer",
                    "inventory_id",
                    "ext_grade",
                    "source_channel",
                    "tgt_enc_iter_impt_platforms",
                    "ext_model_family",
                    "order_no",
                    "line",
                    "inventory_id",
                    "gp",
                    "selling_price",
                    "selling_cost",
                    "invoice_no",
                    "customername",
                ]
            )
        )

        # Log the number of curated columns
        mlflow.log_param("curated_column_count", len(curated_columns))

        # Save the curated columns and the pipeline
        curated_columns_path = op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
        features_transformer_path = op.abspath(op.join(artifacts_folder, "features.joblib"))

        save_pipeline(curated_columns, curated_columns_path)
        save_pipeline(features_transformer, features_transformer_path)

        # Log the paths to the saved artifacts
        mlflow.log_artifact(curated_columns_path)
        mlflow.log_artifact(features_transformer_path)

        
