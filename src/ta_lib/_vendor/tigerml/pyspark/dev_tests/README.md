TigerML pyspark testing
=======================

**Note**

These test cases are not part of CI build and hence are to be explicitly run by the developers before creating a PR. Follow the steps below to perform testing of `tigerml.pyspark` module.

- Activate the `tigerml-pyspark` conda environment (you may have different environment name for pyspark developments, change it accrodingly)

    ```
        $ conda activate tigerml-pyspark
    ```

- Navigate to the tests folder

    ```
        $ cd python/tigerml/pyspark/dev_tests
    ```

- Place your csv files under data folder in dev_tests


- Use the below commands one by one to run the test cases

    ```
        $ pytest -vv test_eda.py
        $ pytest -vv test_model_eval.py
    ```
