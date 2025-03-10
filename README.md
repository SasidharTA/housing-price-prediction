# Sales price prediction

Use regression to predict price of electronic devices

Tip: If you don't have markdown viewer like atom, you can render this on chrome by following [this link](https://imagecomputing.net/damien.rohmer/teaching/general/markdown_viewer/index.html).

# Pre-requisites

* Ensure you have `Miniconda` installed and can be run from your shell. If not, download the installer for your platform here: https://docs.conda.io/en/latest/miniconda.html

     **NOTE**

     * If you already have `Anaconda` installed, go ahead with the further steps, no need to install miniconda.
     * If `conda` cmd is not in your path, you can configure your shell by running `conda init`.


* Ensure you have `git` installed and can be run from your shell

     **NOTE**

     * If you have installed `Git Bash` or `Git Desktop` then the `git` cli is not accessible by default from cmdline.
       If so, you can add the path to `git.exe` to your system path. Here are the paths on a recent setup

```
        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe
```

* Ensure [invoke](http://www.pyinvoke.org/index.html) tool and pyyaml are installed in your `base` `conda` environment. If not, run

```
(base):~$ pip install invoke
(base):~$ pip install pyyaml
```

# Getting started

* Switch to the root folder (i.e. folder containing this file)
* A collection of workflow automation tasks can be seen as follows
    **NOTE**

     * Please make sure there are no spaces in the folder path. Environment setup fails if spaces are present.

```
(base):~/<proj-folder>$ inv -l
```

* To verify pre-requisites, run

```
(base)~/<proj-folder>$ inv debug.check-reqs
```

and check no error messages (`Error: ...`) are printed.


## Environment setup:

### Introduction
* Environment is divided into two sections

    * Core - These are must-have packages and will be set up by default. They are declared in `deploy/pip/ct-core-dev.txt` and the same packages are also declared in `deploy/conda_envs/ct-core-dev.yml`.
    * Addons - These are for specific purposes you can choose to install. Here are the addon options
        * `formatting` - To enforce coding standards in your projects.
        * `documentation` - To auto-generate doc from doc strings and/or create rst style documentation to share documentation online
        * `testing` - To use automated test cases
        * `jupyter` - To run the notebooks. This includes jupyter extensions for spell check, advances formatting.
        * `extras` - there are nice to haves or for pointed usage.
        * `ts` - Install this to work with time series data
        * `pyspark` - Installs pyspark related dependencies in the env.
    * Edit the addons here `deploy/pip/addon-<addon-name>-dev.txt` to suit your need.
    * Each of the packages there have line comments with their purpose. From an installation standpoint extras are treated as addons
* You can edit them to your need. All these packages including addons & extras are curated with versions & tested throughly for acceleration.
* While you can choose, please decide upfront for your project and everyone use the same options.
* Below you can see how to install the core environment & addons separately. However, we strongly recommend to update the core env with the addons packages & extras as needed for your project. This ensures there is only one version of the env file for your project.
* **To run the reference notebooks and production codes, it is recommended to install all addons.**
* Tip: Default name of the env is `ta-lib-dev`. You can change it for your project.
    * For example: to make it as `env-myproject-prod`.
    * Open `tasks.py`
    * Set `ENV_PREFIX = 'env-customer-x'`

#### Setup a development environment:

Run below to install core libraries
```
(base):~/<proj-folder>$ inv dev.setup-env --usecase=<specific usecase>
```

The above command should create a conda python environment named `ta-lib-dev` and install the code in the current repository along with all required dependencies.

`usecase` parameter above is an optional parameter. It takes a value of `tpo` or `mmx` or `ebo` or `rtm` or `reco`.
dev.setup-env in itself will only install the core libs required but when you have to work
with specific use case (e.g MMX or TPO, etc.), one has to install the libraries required for
these specific use cases. So when we provide the `usecase` option, we are specifying that we
want that dependencies for this use case installed in our environment as well.

#### Creating Conda Environment Without Specifying a Usecase:

If you are creating a environment without specifying a usecase, you have the option to set a specific python 
version. In this case the core packages will be installed from `ct-core-dev.txt`.
The below command will create a conda environment with python version `3.9`.

```
(base):~/<proj-folder>$ inv dev.setup-env --python-version=3.9
```
`python-version` parameter above is an optional parameter. By default it is set to `3.10` but it can take values of `3.8`, `3.9` or `3.10`.

#### Creating Conda Environment Specifying a Usecase:
When specifying a usecase, note that the `--python-version` option won't take effect. The environment will be created using the  `ct-core-dev.yml` and
the relevant usecase YAML file. The below command will create a conda environment for specific usecase.

```
(base):~/<proj-folder>$ inv dev.setup-env --usecase=<usecase>
```

Activate the environment first to install other addons. Keep the environment active for all the remaining commands in the manual.
```
(base):~/<proj-folder>$ conda activate ta-lib-dev
```

Install `invoke` and `pyyaml` in this env to be able to install the addons in this environment.
```
(ta-lib-dev):~/<proj-folder>$ pip install invoke
```

Now run all following command to install all the addons. Feel free to customize addons as suggested in the introduction.

```
(ta-lib-dev):~/<proj-folder>$ inv dev.setup-addon --formatting --jupyter --documentation --testing --extras --ts
```

You now should have a standalone conda python environment and installed code in the current repository along with all required dependencies.

* Get the installation info by running
```
(ta-lib-dev):~/<proj-folder>$ inv dev.info
```

* Test your installation by running
```
(ta-lib-dev):~/<proj-folder>$ inv test.val-env --usecase=<specific usecase>
```

We need to specify the usecase to validate the environment for core as well as usecase specific dependencies.

* This will just check the core setup, i.e, the env setup by inv dev.setup-env
* To check the addon installation in the conda env, we check it by specifying the specific addon like
```
(ta-lib-dev):~/<proj-folder>$ inv test.val-env --formatting --jupyter --documentation --testing --extras --ts --pyspark
```
* You can specify which addon's installation you want to check here.

## Manual Environment Setup

If you are facing difficulties setting up the environment using the automated process (i.e., using the `invoke command`) or if the command is not accessible, you can use these manual steps.  This approach is 
particularly useful when troubleshooting or in situations where automated setup is not feasible.
Follow the steps below to manually set up the environment.

### Step 1: Create virtual Environment
```
(base):~/<proj-folder>$ conda create --name <env_name> python=<python_version>
```
Replace `<env_name>` with your specific environment name, and `<python_version>` with the desired python version (e.g., `3.8`, `3.9`, `3.10`).

### Step 2: Activate the Environment
```
(base):~/<proj-folder>$ conda activate <env_name>
```

### Step 3: Install Core Packages
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/ct-core-dev.txt
```

### Step 4: Install the ta_lib editable package
```
(<env_name>):~/<proj-folder>$ pip install -e <path_to_setup.py>
```
if you are in the same level as the `setup.py` file, you can use:
```
(<env_name>):~/<proj-folder>$ pip install -e .
```

### Step 5 (Optional): Install Additional Packages Based on Use Case
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/ct-<usecase>-dev.txt
```
For example, to install packages for `mmx` usecase, use the following command:
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/ct-mmx-dev.txt
```

### Step 6 (Optional): Install Additional Addons
```
(<env_name>):~/<proj-folder>$ pip install -r <path_to_addon_file>
```
For example, to install `jupyter` addons, use the following command:
```
(<env_name>):~/<proj-folder>$ pip install -r deploy/pip/addon-jupyter-dev.txt
```

## Setting Up Environment in Cloud
In a cloud environment invoke commands may not be accessible.
To set up the environment in a cloud setting, you can refer to the following link: [Cloud Environment Setup](https://tigeranalytics-code-templates.readthedocs-hosted.com/en/latest/code_templates/installation_setup.html)

# Launching Jupyter Notebooks

- In order to launch a jupyter notebook locally in the web server, run

    ```
    (ta-lib-dev):~/<proj-folder>$ inv launch.jupyterlab
    ```
     After running the command, type [localhost:8080](localhost:8080) to see the launched JupyterLab.

- The `inv` command has a built-in help facility available for each of the invoke builtins. To use it, type `--help` followed by the command:
    ```
    (ta-lib-dev):~/<proj-folder>$ inv launch.jupyterlab --help
    ```
- On running the ``help`` command, you get to see the different options supported by it.

    ```
    Usage: inv[oke] [--core-opts] launch.jupyterlab [--options] [other tasks here ...]

    Options:
    -a STRING, --password=STRING
    -e STRING, --env=STRING
    -i STRING, --ip=STRING
    -o INT, --port=INT
    -p STRING, --platform=STRING
    -t STRING, --token=STRING
    ```

# Frequently Asked Questions

The FAQ for code templates during setting up, testing, development and adoption phases are available
[here](https://tigeranalytics-code-templates.readthedocs-hosted.com/en/latest/faq.html)