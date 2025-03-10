============
Organization
============


Project Folder Structure
========================

The directories in the project are organized in the following for easy maintainability and faster productionizing :


.. _config_folder:

.. figure:: ../images/project_structure.png
   :width: 200px
   :height: 200px
   :alt: Project Structure
   :name: Project Structure
   :align: center
   


The description of the folder structure is as follows.
    
1. ``artifacts`` - Files like Model pickle files, transformers are stored here which are used for model validation/scoring. This wouldn't be needed if the storage is on cloud. Please note that changes to this folder should not be updated in code repository.

2. ``data`` - This directory forms the local repository for storing input or output data used for EDA, modeling, pre processing or post processing steps. This wouldn't be needed if the storage is on cloud. Please note that changes to this folder should not be updated in code repository.

3. ``deploy`` - This directory contains files to replicate or recreate the development environment such as  yml files, pip requirements, conda lock files etc..

4. ``docs`` - Project documentation in ReST format. 

5. ``logs`` - Folder to store log files. Development / production logs can be saved here. Please note that changes to this folder should not be updated in code repository.

6. ``notebooks`` - This is the development location. Developer should create a folder which will ask as his/her workspace. Jupyter notebooks, scripts, configurations will be part of this directory either divided by usecases, method of development like R or pyspark or python etc.

7. ``production`` - Production scripts that are either put on schedule or executed manually will be part of this directory. This directory has codes from ``notebooks`` folder which essentially does jobs like data ingestion, model training, model scoring etc.

8. ``src\ta_lib`` - All reusable code is goes here which has all the common utilities for ``notebooks`` and ``production`` folders. This source code includes programs for IO read and write for file systems in project like s3, gcs or dbfs and utilities for data processing, feature transformations, reports for eda, model evaluation etc..
