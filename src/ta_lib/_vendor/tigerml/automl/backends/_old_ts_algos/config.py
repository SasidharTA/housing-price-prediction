# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 02:43:13 2018.

@author: ranjith.a
"""

import json
import os

with open("conf.json") as json_data_file:
    data = json.load(json_data_file)

algorithms = data["default"]["algorithms"]
data_location = data["default"]["data_location"]
idv_variable_names = data["default"]["idv_variable_names"]
#
hyperparams_simpleexponentialsmoothing = data["hyperparams"][
    "SimpleExponentialSmoothing"
]
hyperparams_exponentialsmoothingholt = data["hyperparams"]["ExponentialSmoothingHolt"]
hyperparams_exponentialsmoothingholtwinters = data["hyperparams"][
    "ExponentialSmoothingHoltWinters"
]
hyperparams_sarimax = data["hyperparams"]["SARIMAX"]
hyperparams_lstmseqtoseqmultivariate = data["hyperparams"]["LSTMSeqToSeqMultivariate"]
#
perf_metrics = data["default"]["performance_metrics"]
dv_variable_name = data["default"]["dv_variable_name"]

# creating an output folder for model results and saved models
code_output_path = os.getcwd()
code_output_path = code_output_path + "/Output_directory/"
if not os.path.exists(code_output_path):
    os.makedirs(code_output_path)
