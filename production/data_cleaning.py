import numpy as np
import pandas as pd
import mlflow
import mlflow.pandas
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)
from scripts import binned_selling_price


@register_processor("data-cleaning", "product")
def clean_product_table(context, params):
    """Clean the ``PRODUCT`` data table."""
    input_dataset = "raw/product"
    output_dataset = "cleaned/product"

    # Start MLflow run
    with mlflow.start_run(run_name="Clean Product Table") as run:
        # Load dataset
        product_df = load_dataset(context, input_dataset)

        product_df_clean = (
            product_df
            .passthrough()
            .transform_columns(
                product_df.columns.to_list(), string_cleaning, elementwise=False
            )
            .replace({"": np.NaN})
            .coalesce(["color", "Ext_Color"], "color", delete_columns=True)
            .coalesce(["MemorySize", "Ext_memorySize"], "memory_size", delete_columns=True)
            .remove_duplicate_rows(col_names=["SKU"], keep_first=True)
            .clean_names(case_type="snake")
        )

        # Log the number of rows and columns before and after cleaning
        mlflow.log_param("initial_row_count", len(product_df))
        mlflow.log_param("initial_column_count", len(product_df.columns))
        mlflow.log_param("cleaned_row_count", len(product_df_clean))
        mlflow.log_param("cleaned_column_count", len(product_df_clean.columns))

        # Log cleaned dataset as artifact
        mlflow.log_artifact("cleaned_product.csv", product_df_clean.to_csv(index=False))

        # Save the cleaned dataset
        save_dataset(context, product_df_clean, output_dataset)

        return product_df_clean


@register_processor("data-cleaning", "orders")
def clean_order_table(context, params):
    """Clean the ``ORDER`` data table."""
    input_dataset = "raw/orders"
    output_dataset = "cleaned/orders"

    # Start MLflow run
    with mlflow.start_run(run_name="Clean Order Table") as run:
        # Load dataset
        orders_df = load_dataset(context, input_dataset)

        # String cleaning on specific columns
        str_cols = list(
            set(orders_df.select_dtypes("object").columns.to_list())
            - set(["Quantity", "InvoiceNo", "Orderno", "LedgerDate"])
        )
        orders_df_clean = (
            orders_df
            .change_type(["Quantity", "InvoiceNo", "Orderno"], np.int64)
            .to_datetime("LedgerDate", format="%d/%m/%Y")
            .transform_columns(str_cols, string_cleaning, elementwise=False)
            .clean_names(case_type="snake").rename_columns({"orderno": "order_no"})
        )

        # Log initial and cleaned dataset sizes
        mlflow.log_param("initial_row_count", len(orders_df))
        mlflow.log_param("initial_column_count", len(orders_df.columns))
        mlflow.log_param("cleaned_row_count", len(orders_df_clean))
        mlflow.log_param("cleaned_column_count", len(orders_df_clean.columns))

        # Log cleaned dataset as artifact
        mlflow.log_artifact("cleaned_orders.csv", orders_df_clean.to_csv(index=False))

        # Save the cleaned dataset
        save_dataset(context, orders_df_clean, output_dataset)
        return orders_df_clean


@register_processor("data-cleaning", "sales")
def clean_sales_table(context, params):
    """Clean the ``SALES`` data table."""
    input_product_ds = "cleaned/product"
    input_orders_ds = "cleaned/orders"
    output_dataset = "cleaned/sales"

    # Start MLflow run
    with mlflow.start_run(run_name="Clean Sales Table") as run:
        # Load datasets
        product_df = load_dataset(context, input_product_ds)
        orders_df = load_dataset(context, input_orders_ds)

        # Merge sales data
        sales_df_clean = orders_df.merge(product_df, how="inner", on="sku")

        # Log dataset details
        mlflow.log_param("product_row_count", len(product_df))
        mlflow.log_param("orders_row_count", len(orders_df))
        mlflow.log_param("sales_row_count", len(sales_df_clean))

        # Log cleaned sales dataset as artifact
        mlflow.log_artifact("cleaned_sales.csv", sales_df_clean.to_csv(index=False))

        # Save the cleaned dataset
        save_dataset(context, sales_df_clean, output_dataset)

        return sales_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``SALES`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/sales"
    output_train_features = "train/sales/features"
    output_train_target = "train/sales/target"
    output_test_features = "test/sales/features"
    output_test_target = "test/sales/target"

    # Start MLflow run
    with mlflow.start_run(run_name="Create Train/Test Datasets") as run:
        # Load dataset
        sales_df_processed = load_dataset(context, input_dataset)

        # Feature engineering
        cust_details = (
            sales_df_processed.groupby(["customername"])
            .agg({"ledger_date": "min"})
            .reset_index()
        )
        cust_details.columns = ["customername", "ledger_date"]
        cust_details["first_time_customer"] = 1
        sales_df_processed = sales_df_processed.merge(
            cust_details, on=["customername", "ledger_date"], how="left"
        )
        sales_df_processed["first_time_customer"].fillna(0, inplace=True)

        # Days since last purchase
        sales_df_processed.sort_values("ledger_date", inplace=True)
        sales_df_processed["days_since_last_purchase"] = (
            sales_df_processed.groupby("customername")["ledger_date"]
            .diff()
            .dt.days.fillna(0, downcast="infer")
        )

        # Log feature engineering details
        mlflow.log_param("first_time_customer_feature", "added")
        mlflow.log_param("days_since_last_purchase_feature", "added")

        # Split the data
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=params["test_size"], random_state=context.random_seed
        )
        sales_df_train, sales_df_test = custom_train_test_split(
            sales_df_processed, splitter, by=binned_selling_price
        )

        # Split train dataset into features and target
        target_col = params["target"]
        train_X, train_y = (
            sales_df_train
            .get_features_targets(target_column_names=target_col)
        )

        # Log train set details
        mlflow.log_param("train_row_count", len(train_X))
        mlflow.log_param("train_column_count", len(train_X.columns))

        # Save train dataset
        save_dataset(context, train_X, output_train_features)
        save_dataset(context, train_y, output_train_target)

        # Split test dataset into features and target
        test_X, test_y = (
            sales_df_test
            .get_features_targets(target_column_names=target_col)
        )

        # Log test set details
        mlflow.log_param("test_row_count", len(test_X))
        mlflow.log_param("test_column_count", len(test_X.columns))

        # Save test dataset
        save_dataset(context, test_X, output_test_features)
        save_dataset(context, test_y, output_test_target)

