import mlflow
import os
from ta_lib.core.api import (
    load_dataset,
    save_dataset,
    save_pipeline,
    register_processor
)
from ta_lib.data_processing.api import Outlier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from ta_lib.regression.api import SKLStatsmodelOLS
from ta_lib.core.api import (get_dataframe,
                             get_feature_names_from_column_transformer,
                             get_package_path, hash_object,
                             load_pipeline, DEFAULT_ARTIFACTS_PATH)
from data_cleaning import clean_product_table, clean_order_table, clean_sales_table, create_training_datasets
import pandas as pd

# Example function for data cleaning
def clean_product_table(context, params):
    product_df = load_dataset(context, "raw/product")
    product_df_clean = product_df.replace({"": pd.NaT}).drop_duplicates(subset=["SKU"]).fillna("")
    save_dataset(context, product_df_clean, "cleaned/product")
    return product_df_clean

# Example function for feature engineering
def transform_features(context, params):
    train_X = load_dataset(context, "train/sales/features")
    train_y = load_dataset(context, "train/sales/target")
    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Handle outliers and imputation
    outlier_transformer = Outlier(method=params.get("outliers_method", "iqr"))
    train_X = outlier_transformer.fit_transform(train_X)

    # Transform the features using a pipeline (encoding and imputation)
    feature_transformer = Pipeline([
        ('target_encode', TargetEncoder()),
        ('simple_impute', SimpleImputer(strategy='most_frequent'))
    ])

    train_X = feature_transformer.fit_transform(train_X, train_y)
    save_pipeline(feature_transformer, "feature_transformer")
    save_dataset(context, train_X, "transformed_features")
    return train_X, train_y

# Model Training Job
def train_model(context, params):
    train_X = load_dataset(context, "transformed_features")
    train_y = load_dataset(context, "train/sales/target")

    # Create a regression model pipeline
    reg_pipeline = Pipeline([
        ('estimator', SKLStatsmodelOLS())
    ])

    reg_pipeline.fit(train_X, train_y)

    save_pipeline(reg_pipeline, "trained_model_pipeline")
    mlflow.sklearn.log_model(reg_pipeline, "model")
    return reg_pipeline

# Scoring Job
def score_model(context, params):
    test_X = load_dataset(context, "test/sales/features")
    test_y = load_dataset(context, "test/sales/target")

    # Load trained model
    model_pipeline = load_pipeline("trained_model_pipeline")

    # Score the model
    test_X["predictions"] = model_pipeline.predict(test_X)
    save_dataset(context, test_X, "scored_predictions")
    return test_X

experiment_name = "house_price_experiment"  # Give your experiment a name
mlflow.set_experiment(experiment_name)

# Main Job for House Price Prediction
@register_processor("house-price-prediction", "main-job")
def house_price_prediction(context, params):
    with mlflow.start_run(run_name="House Price Prediction Pipeline"):
        mlflow.log_param("main_step", "House Price Prediction")

        # 1. Data Cleaning (Child Job)
        with mlflow.start_run(run_name="Data Cleaning", nested=True):
            clean_product_table(context, params)
            mlflow.log_param("step", "Product Data Cleaned")
            clean_order_table(context, params)
            mlflow.log_param("step", "Order Data Cleaned")
            clean_sales_table(context, params)
            mlflow.log_param("step", "Sales Data Cleaned")
            create_training_datasets(context, params)
            mlflow.log_param("step", "Train/Test Split Completed")

        # 2. Feature Engineering (Child Job)
        with mlflow.start_run(run_name="Feature Engineering", nested=True):
            transform_features(context, params)
            mlflow.log_param("step", "Feature Transformation Completed")

        # 3. Model Training (Child Job)
        with mlflow.start_run(run_name="Model Training", nested=True):
            trained_model = train_model(context, params)
            mlflow.log_param("step", "Model Training Completed")

        # 4. Model Scoring (Child Job)
        with mlflow.start_run(run_name="Model Scoring", nested=True):
            scored_predictions = score_model(context, params)
            mlflow.log_param("step", "Model Scored")

        mlflow.log_param("main_step", "House Price Prediction Pipeline Completed")

# Main function to run all jobs
if __name__ == "__main__":
    # Define params for the main job and child jobs
    params = {
        "outliers_method": "iqr",  # Example parameter for feature engineering
        # Add more parameters as required
    }

    # Execute the main house price prediction job
    house_price_prediction(context=None, params=params)
