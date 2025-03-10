# AutoML
- This AutoML module automates the process of selecting best models. 
- This module helps with supervised learning problems - **Classification (binary/multiclass)** and **Regression**.
- Pipelines that the tool evaluates includes feature processing, feature selection, multiple ML algorithms, and corresponding hyper-parameters.
- It is based on TPOT Python package (https://epistasislab.github.io/tpot/) that optimizes machine learning pipelines using Genetic Programming.

# AutoML set-up in personal workstation
**Requirements**
- You need Python 3.6 and above to use this tool
- Download the tigerml package (.whl file) and install it with the command below. Replace the placeholder in the command with the path to the .whl file in your system. 
```
    pip install <path__to__whl__file>
```
**Steps to verify**
- Open python3 console
- run the following command
```
from tigerml.automl import AutoML
```
- If this works without errors, you are all set (Ignore import warnings)

# AutoML API:
| **Function**                   | **Arguments** | **Description**                                                                         |
| ------------------------------ | ------------- | --------------------------------------------------------------------------------------- |
| AutoML() (initiation)          | name, task, x_train, y_train, x_test, y_test, random_state, search_space, scoring_func, cv_folds       | This function prepares data as required for modeling                           |
| AutoML.prep_data()           | data, dv_name, train_size, remove_na, impute_num_na, impute_cat_na       | Takes the complete dataset and prepares train and test splits after handling the missing values                |
| AutoML.fit()           |       | Runs the pipeline to search the model and hyperparameter space given to optimize for the scoring metric |
| AutoML.get_report()           | no_of_pipelines      | This function outputs performance metrics for the top *n* best models |


## AutoML object:

**Description**

Create the AutoML object specific to the task and dataset

**Arguments**

name: Name your run (optional) default: ''

task: Task to be done (classification/regression) default: 'classification'

x_train: pandas.DataFrame - Independent variables in train data (optional)

y_train: pandas.DataFrame - Dependent variable in train data (optional)

x_test: pandas.DataFrame - Independent variables in test data (optional)

y_test: pandas.DataFrame - Dependent variables in test data (optional)
 
random_state: Random state value to be used in .data_prep and .fit functions

search_space: Set the search space for the AutoML pipeline (default values - default, light, mdr, sparse, imbalanced)

scoring_func: the evaluation criteria to optimize the pipelines
values for classification: ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc', 'log_loss', 'f1_score', 'balanced_accuracy']
values for regression: ['neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error', 'r2', 'MAPE', 'WMAPE']

cv_folds: no of folds for cross validation

preprocessing: bool - True/False to indicate if preprocessing can be included in model pipeline

selection: bool - True/False to indicate if feature selection can be included in model pipeline

modeling: dict - dict object with all model algos and hyperparameters for each in the search space


# Examples
Refer to automl_sample.py file in the attachments to see sample code to run classification and regression models

# AutoML set-up on AWS
- AutoML is a resource intensive application and is recommended to be run using cloud resources
- When you request IT team to set-up AWS environment for your project, request them to include Tiger AutoML AMI (Amazon Machine Image)
- Once you ssh into the server, you can install tigerml package as mentioned above and you'll be ready to run AutoML

**If you are in Windows:**

You need a couple of applications, PuTTY and WinSCP, to use AWS instance

## PuTTY
PuTTY is a common utility to connect to remote linux servers

**Pre-requisties**
- You should have putty.exe. You can download it [here](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html). Get 64 bit putty.exe from "Alternative binary files" section.
- Get the Public IP address & ppk key file for your instance from IT Helpdesk

**Setup**
- Double click on putty. You should get something like the below image.
![](https://github.com/tigerrepository/TA_LAB/blob/master/AutoML/examples/docs/02_putty_session.png)
- In the Host Name text box type ubuntu@\<yourip>.
- On the left panel navigate to Connection -> SSH -> Auth and browse for ppk file. Refer below image.
![](https://github.com/tigerrepository/TA_LAB/blob/master/AutoML/examples/docs/03_putty_key.jpg)
- On the left panel navigate to Session.
- In the text box below Saved Sessions, give it a name like AutoML and click save.

## WinSCP
WinSCP is a common utility to copy files to and from remote linux servers

**Pre-requisties**
- You can download the portalbe version of WinSCP from [here](https://winscp.net/eng/downloads.php).
- Get the Public IP address & ppk key file for your instance from IT Helpdesk

**Setup**
- Double click on WinSCP.exe (~18 MB file). Refer to image below.
![](https://github.com/tigerrepository/TA_LAB/blob/master/AutoML/examples/docs/04_WINSCP_session.png)
- In the Host Name text box type \<yourip>.
- In the User name text box type ubuntu.
- Click on advanced. Then navigate to SSH -> Authentication. Refer below image.
- In the box below private key, browse for ppk file
![](https://github.com/tigerrepository/TA_LAB/blob/master/AutoML/examples/docs/05_WINSCP_key.png)
- Click ok on the advanced tab. Then click Save. In the popup window give it an appropriate name.


# Running AutoML on AWS
- Consider using Compute Optimized instance - c5.4xlarge 16 (vCPUs) and 32 (Memory GB)
- Once you log into the AWS machine, use **python3** to run your codes.
- Create a folder to save the intermediate best pipelines. This will have intermediate results when your run terminates abruptly. Use 'int_result_folder' parameter from 'automl.perform_model_optimization' function.
- Launch AutoML Python script using following code
```
	nohup python3 automl_sample.py > automl_sample.log  2>&1 &
```
- automl_sample.log outputs the progress of the optimization process to the log file. To check the status use the below command
```
	tail automl_sample.log
```
- To pool the status to console, use the following command
```
	tail -f automl_sample.log
```

# Additional details about TPOT
- Check https://epistasislab.github.io/tpot/api/ for additional details about what other parameters you can control in your run.
- This also has information about default values set for each of these parameters.  

# Guidelines
- TPOT runs take lot of time to run. Plan your runs accordingly.
  - Number of models that TPOT runs is equal to *(population_size + population_size * num_generations) * num_cv*. Estimate the total time that TPOT likely takes based on a single model run.
  - TPOT by default only analyzes pipelines that take less than 5 mins. If your dataset is large, consider increasing *max_eval_time_mins*  
- TPOT runs typically break because there is not enough memory (RAM). Monitor your RAM usage and if you are consistently touching 90%, consider increasing the size of RAM.
- Factors that impact usage of RAM
  - Size of the dataset
  - Large number of rows
  - Lot more continuous variables than categorical/binary variables
  - Usage of pre-processing modules
- Apart from increasing the size of RAM, you can try following options if your runs are failing
  - Remove date variables
  - Include only a subset of pre-processing modules (sklearn.preprocessing.Binarizer, sklearn.preprocessing.MinMaxScaler, sklearn.preprocessing.Normalizer, sklearn.decomposition.PCA, tpot.builtins.ZeroCount)
  - Exclude all pre-processing modules
- Sometimes when the population size is low, runs may fail because TPOT is not able to select enough feasible solutions. Try increasing the population size in such cases.
- In highly imbalanced datasets (one class is about 1%), you should consider using class weights parameter or other relevant parameters that address class imbalance. You could either use 'balanced' option or provide a range of parameter values to evaluate. See the examples/Config_ClassImbalance.py file.
- We evaluated TPOT results against H2O Driverless AI (another popular AutoML tool) on two dimensions - model performance and computation time. And in majority of cases TPOT performed on par with H2O. You can find the comparison results in examples/Results_summary_final. This file also has details about potential issues that you will encounter and how you can address them. 

