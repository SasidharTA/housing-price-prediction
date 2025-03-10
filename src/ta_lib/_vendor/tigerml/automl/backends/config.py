# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 02:43:13 2018.

@author: kapil.kumar3290
"""

import copy
import json
import os

json_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(json_path, "conf.json")) as json_data_file:
    data = json.load(json_data_file)

dafault_config = copy.deepcopy(data)

# algorithms = dafault_config["algorithms"]
# validation_hyperparams = dafault_config["validation_hyperparams"]
# model_hyperparams = dafault_config["model_hyperparams"]
# perf_metrics = dafault_config["performance_metrics"]


# creating an output folder for model results and saved models
code_output_path = os.getcwd()
code_output_path = code_output_path + "/Output_directory/"
if not os.path.exists(code_output_path):
    os.makedirs(code_output_path)
