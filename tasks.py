import fnmatch
import os
import os.path as op
import platform
import shutil
import tempfile
import time
import yaml
from contextlib import contextmanager
from invoke import Collection, UnexpectedExit, task

# Some default values
PACKAGE_NAME = "ta_lib"
ENV_PREFIX = "ta-lib"
ENV_PREFIX_PYSPARK = "ta-lib-pyspark"
NUM_RETRIES = 10
SLEEP_TIME = 1

OS = platform.system().lower()
ARCH = platform.architecture()[0][:2]
PLATFORM = f"{OS}-cpu-{ARCH}"
DEV_ENV = "dev"  # one of ['dev', 'run', 'test']

HERE = op.dirname(op.abspath(__file__))
SOURCE_FOLDER = op.join(HERE, "src", PACKAGE_NAME)
TESTS_FOLDER = op.join(HERE, "tests")
CONDA_ENV_FOLDER = op.join(HERE, "deploy", "conda_envs")
PIP_REQ_FOLDER = op.join(HERE, "deploy", "pip")
PYSPARK_ENV_FOLDER = op.join(HERE, "deploy", "pyspark")
NOTEBOOK_FOLDER = op.join(HERE, "notebooks", "tests")

TESTS_USECASE = [
    "core",
    "ebo",
]

_TASK_COLLECTIONS = []

addon_dict = {
    "documentation": "addon-documentation",
    "formatting": "addon-code_format",
    "extras": "addon-extras",
    "jupyter": "addon-jupyter",
    "testing": "addon-testing",
    "ts": "addon-ts",
    "tareg": "addon-tareg"
}

# ---------
# Utilities
# ---------


def _get_env_name(env):
    return f"{ENV_PREFIX}-{env}"


def _get_env_name_pyspark(env):
    return f"{ENV_PREFIX_PYSPARK}-{env}"


def _change_permissions_recursive(path, mode):
    for root, dirs, files in os.walk(path, topdown=False):
        for _dir in [os.path.join(root, d) for d in dirs]:
            os.chmod(_dir, mode)
        for _file in [os.path.join(root, f) for f in files]:
            os.chmod(_file, mode)


def _clean_rmtree(path):
    for _try in range(NUM_RETRIES):
        try:
            _change_permissions_recursive(path, 0o777)
            shutil.rmtree(path)
        except Exception as e:
            time.sleep(SLEEP_TIME)
            print(f""" "{path}" Remove failed with error {e}:: Retrying ..""")
            continue
        print(f""" "{path}" Remove Success""")
        break


@contextmanager
def py_env(c, env_name):
    """Activate a python env while the context is active."""

    # FIXME: This works but takes a few seconds. Perhaps find a few to robustly
    # find the python path and just set the PATH variable ?
    if OS == "windows":
        # we assume conda binary is in path
        cmd = f"conda activate {env_name}"
    else:
        cmd = f'eval "$(conda shell.bash hook)" && conda activate {env_name}'
    with c.prefix(cmd):
        yield


def _create_task_collection(name, *tasks):
    """Construct a Collection object."""
    coll = Collection(name)
    for task_ in tasks:
        coll.add_task(task_)
    _TASK_COLLECTIONS.append(coll)
    return coll


def _create_root_task_collection():
    return Collection(*_TASK_COLLECTIONS)

def get_package_version(source_folder):
    with open(op.join(source_folder, "version.py")) as fp:
        ns = {}
        exec(fp.read(), ns, ns)
        package_version = ns["version"]
        return package_version


# ---------
# debug tasks
# ---------


@task(name="check-reqs")
def check_setup_prerequisites(c):
    _failed = []

    # check the folder has no spaces in the path
    # if " " in HERE:
    #     raise RuntimeError("The path to the current folder has whitespaces in it.")

    for binary in ["git", "conda"]:
        try:
            out = c.run(f"{binary} --version", hide="out")
        except UnexpectedExit:
            print(
                f"ERROR: Failed to find `{binary}` in path. "
                "See `pre-requisites` section in `README.md` for some pointers."
            )
            _failed.append(binary)
        else:
            print(f"SUCCESS: Found `{binary}` in path : {out.stdout}")

    # FIXME: Figure out colored output (on windows) and make the output easier
    # to read/interpret.
    # for now, make a splash when failing and make it obvious.
    if _failed:
        raise RuntimeError(f"Failed to find the following binaries in path : {_failed}")


_create_task_collection("debug", check_setup_prerequisites)


# Dev tasks
# ---------
@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``{windows|linux}-{cpu|gpu}-{64|32}``"
        ),
        "env": "Specifies the environment type. Must be one of ``{dev|test|run}``",
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
    }
)
def setup_env_legacy(c, platform=PLATFORM, env=DEV_ENV, force=False):
    """Help in setup of a new development environment.

    Creates a new conda environment with the dependencies specified in the file
    ``env/{platform}-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """

    # run pre-checks
    check_setup_prerequisites(c)

    # FIXME: We might want to split the env.yml file into multiple
    # files: run.yml, build.yml, test.yml and support combining them for
    # different environments
    force_flag = "" if not force else "--force"

    env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"{platform}-{env}.lock"))
    req_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"requirements-{platform}-{env}.txt")
    )
    req_flag = True
    if not op.isfile(env_file):
        raise ValueError(f"""The conda env file is not found : "{env_file}" """)
    if not op.isfile(req_file):
        req_flag = False
    env_name = _get_env_name(platform, env)

    out = c.run(f"""conda create --name {env_name} --file "{env_file}"  {force_flag} -y""")

    # check for jupyterlab
    with open(env_file, "r") as fp:
        env_cfg = fp.read()

    # installing jupyter lab extensions
    extensions_file = op.abspath(op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml"))
    with open(extensions_file) as fp:
        extensions = yaml.safe_load(fp)

    # install the code-template modules
    with py_env(c, env_name):

        # install pip requirements
        if req_flag:
            c.run(f"""python -m pip install -r "{req_file}" --no-deps""")

        # install the current package
        c.run(f"""python -m pip install -e "{HERE}" """)

        is_jupyter = False
        if "jupyterlab-" in env_cfg:
            is_jupyter = True

        if is_jupyter:
            # install jupyterlab extensions
            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(
                    f"jupyter labextension install --no-build {extn_name}",
                )

            out = c.run("jupyter lab build")

    # FIXME: create default folders that are expected. these need to be handled
    # when converting to cookiecutter templates
    os.makedirs(op.join(HERE, "logs"), exist_ok=True)
    os.makedirs(op.join(HERE, "docs", "build", "html"), exist_ok=True)
    os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)
    os.makedirs(op.join(HERE, "data"), exist_ok=True)

def _jupyterlab_install(c, env_name, env_file):
    # check for jupyterlab
    with open(env_file, "r") as fp:
        env_cfg = fp.read()

    # installing jupyter lab extensions
    extensions_file = op.abspath(op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml"))
    with open(extensions_file) as fp:
        extensions = yaml.safe_load(fp)

    with py_env(c, env_name):

        is_jupyter = False
        if "jupyterlab-" in env_cfg:
            is_jupyter = True

        if is_jupyter:
            # install jupyterlab extensions
            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(
                    f"jupyter labextension install --no-build {extn_name}",
                )

            out = c.run("jupyter lab build")

def _setup_env_common(c, env_name, platform=PLATFORM, env=DEV_ENV, force=False, python_version="3.10"):
    """
    Set up a Conda environment and install packages using pip. Called when no usecase is specified.
    """
    force_flag = "" if not force else "--yes"

    env_file = op.abspath(op.join(PIP_REQ_FOLDER, f"ct-core-{env}.txt"))

    if not op.isfile(env_file):
        raise ValueError(f"""The conda env file is not found : "{env_file}" """)

    out = c.run(f"""conda create --name {env_name} python={python_version}  {force_flag}""")


    # install the code-template modules
    with py_env(c, env_name):

        # install the current package
        c.run(f"""pip install -r "{env_file}"  """)
        c.run(f"""python -m pip install -e "{HERE}" """)
        c.run(f"""echo "To activate this env use: conda activate {env_name}" """)

    # FIXME: create default folders that are expected. these need to be handled
    # when converting to cookiecutter templates
    os.makedirs(op.join(HERE, "logs"), exist_ok=True)
    os.makedirs(op.join(HERE, "docs", "build", "html"), exist_ok=True)
    os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)
    os.makedirs(op.join(HERE, "data"), exist_ok=True)


def _setup_env_common_usecase(c, env_name, platform=PLATFORM, env=DEV_ENV, force=False):
    """
    Set up a Conda environment for the specific use case, utilizing `conda env create` and `pip install` 
    when the usecase parameter is used in set_env.
    """
    force_flag = "" if not force else "--force"

    env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-core-{env}.yml"))

    if not op.isfile(env_file):
        raise ValueError(f"""The conda env file is not found : "{env_file}" """)

    out = c.run(f"""conda env create --name {env_name} --file "{env_file}" {force_flag}""")

    # install the code-template modules
    with py_env(c, env_name):

        # install the current package
        c.run(f"""python -m pip install -e "{HERE}" """)
    
    # Commented out below code as it was unnecessary and slowing down environment creation.
    # _jupyterlab_install(c, env_name, env_file)

    # FIXME: create default folders that are expected. these need to be handled
    # when converting to cookiecutter templates
    os.makedirs(op.join(HERE, "logs"), exist_ok=True)
    os.makedirs(op.join(HERE, "docs", "build", "html"), exist_ok=True)
    os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)
    os.makedirs(op.join(HERE, "data"), exist_ok=True)

@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``ct-core``"
        ),
        "env": "Specifies the environment type. Must be one of ``{dev|test|run}``",
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
        "python_version": "Specifies the python version. Must be one of ``{3.8|3.9|3.10}``",
    }
)
def setup_env(c, platform=PLATFORM, env=DEV_ENV, force=False, usecase=None, python_version="3.10"):
    """Help in setup of a new development environment.

    Creates a new conda environment with the dependencies specified in the file
    ``deploy/conda_envs/ct-core-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """

    # run pre-checks
    check_setup_prerequisites(c)

    # FIXME: We might want to split the env.yml file into multiple
    # files: run.yml, build.yml, test.yml and support combining them for
    # different environments

    usecase_file = None
    if usecase:
        if usecase=="tpo":
            usecase_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-tpo-{env}.yml"))
        elif usecase=="mmx":
            usecase_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-mmx-{env}.yml"))
        elif usecase=="ebo":
            usecase_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-ebo-{env}.yml"))
        elif usecase=="rtm":
            usecase_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-rtm-{env}.yml"))
        elif usecase=="reco":
            usecase_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-reco-{env}.yml"))
        else:
            raise FileNotFoundError(
                "This is not a valid usecase. Valid usecases -> tpo or rtm or mmx or ebo")

    env_name = _get_env_name(env)
   
    if usecase:
        _setup_env_common_usecase(c, env_name, platform=platform, env=env, force=force)
        with py_env(c, env_name):

        # install the current package
            out_upd = c.run(f"""conda env update --name {env_name} --file "{usecase_file}" """)
    else:
        _setup_env_common(c, env_name, platform=platform, env=env, force=force, python_version=python_version)
        

@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``ct-core``"
        ),
        "env": "Specifies the environment type. Must be one of ``{dev|test|run}``",
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
        "python_version": "Specifies the python version. Must be one of ``{3.8|3.9|3.10}``",
    }
)
def setup_env_pyspark(c, platform=PLATFORM, env=DEV_ENV, force=False, python_version="3.10"):
    """Help in setup of a new development pyspark environment.

    Creates a new conda environment with the dependencies specified in the file
    ``deploy/conda_envs/{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """

    # run pre-checks
    check_setup_prerequisites(c)

    # FIXME: We might want to split the env.yml file into multiple
    # files: run.yml, build.yml, test.yml and support combining them for
    # different environments
    force_flag = "" if not force else "--force"

    env_file = op.abspath(op.join(PIP_REQ_FOLDER, f"ct-core-{env}.txt"))
    usecase_file = op.abspath(op.join(PIP_REQ_FOLDER, f"ct-pyspark-{env}.txt"))

    # req_file = op.abspath(
    #     op.join(
    #         PYSPARK_ENV_FOLDER,
    #         f"requirements-{env}.txt"
    #     )
    # )
    # req_flag = True
    if not op.isfile(env_file):
        raise ValueError(f"""The conda env file is not found : "{env_file}" """)
    # if not op.isfile(req_file):
    #     req_flag = False
    env_name = _get_env_name_pyspark(env)
    # print(f"env_name is {env_name} \n env_file is {env_file}")  # TODO: Delete this

    _setup_env_common(c, env_name, platform=platform, env=env, force=force, python_version=python_version)
    
    with py_env(c, env_name):
        
        c.run(
            f"""pip install -r "{usecase_file}" """
        )
    # out_upd = c.run(f"""conda env update --name {env_name} --file "{usecase_file}" """)
    # out = c.run(f"conda env create -f {env_file}  {force_flag}")


def _addon_file_paths(platform, env, addon_list):
    addon_file_list = []
    for addon in addon_list:
        addon_path = op.abspath(
            op.join(PIP_REQ_FOLDER, f"{addon}-{platform}-{env}.txt")
        )
        addon_path_without_platform = op.join(PIP_REQ_FOLDER, f"{addon}-{env}.txt")
        if os.path.exists(addon_path):
            addon_file_list.append(addon_path)
        elif os.path.exists(addon_path_without_platform):
            addon_file_list.append(addon_path_without_platform)
        else:
            raise FileNotFoundError(f"""The file for {addon} doesn't exist in "{PIP_REQ_FOLDER}" folder""")


    return addon_file_list

def _addon_update_env(c, addon_file, env_name):
    with py_env(c, env_name):
        c.run(
            f"""pip install -r "{addon_file}" """
        )
    if "documentation" in addon_file:
        os.makedirs(op.join(HERE, "docs/build"), exist_ok=True)
        os.makedirs(op.join(HERE, "docs/source"), exist_ok=True)
        
    # Commented out below code as it was unnecessary
    # jupyterlab is enough to run the notebooks, jupyter extension is not required.
    
    # if "jupyter" in addon_file:
    #     extensions_file = op.abspath(
    #         op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
    #     )
    #     with open(extensions_file) as fp:
    #         extensions = yaml.safe_load(fp)

    #     with py_env(c, env_name):
    #         for extension in extensions["extensions"]:
    #             extn_name = "@{channel}/{name}@{version}".format(**extension)
    #             c.run(f"jupyter labextension install --no-build {extn_name}",)

    #         out = c.run("jupyter lab build")


    if "extras" in addon_file:
        os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)

def _setup_addon_common(c,
    env_name,
    platform=PLATFORM,
    env=DEV_ENV,
    all=False,
    documentation=False,
    testing=False,
    formatting=False,
    jupyter=False,
    extras=False,
    ts=False,
    tareg=False,
    ):

    addon_list = []
    if documentation or all:
        addon_list.append(addon_dict["documentation"])
    if testing or all:
        addon_list.append(addon_dict["testing"])
    if formatting or all:
        addon_list.append(addon_dict["formatting"])
    if jupyter or all:
        addon_list.append(addon_dict["jupyter"])
    if extras:
        addon_list.append(addon_dict["extras"])
    if ts or all:
        addon_list.append(addon_dict["ts"])
    if tareg or all:
        addon_list.append(addon_dict["tareg"])

    addon_file_list = _addon_file_paths(platform=platform, env=env, addon_list=addon_list)

    for addon_file in addon_file_list:
        _addon_update_env(c, addon_file=addon_file, env_name=env_name)

    # addons_documentation = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-documentation-{env}.yml")
    # )
    # addons_testing = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-testing-{env}.yml")
    # )
    # addons_formatting = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-code_format-{env}.yml")
    # )
    # addons_jupyter = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-jupyter-{env}.yml")
    # )
    # addons_extras = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-extras-{env}.yml")
    # )
    # addons_ts = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-ts-{env}.yml")
    # )
    # addons_pyspark = op.abspath(
    #     op.join(CONDA_ENV_FOLDER, f"addon-pyspark-{env}.yml")
    # )
    # with py_env(c, env_name):
    #     if documentation:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_documentation}"
    #         )
    #         os.makedirs(op.join(HERE, "docs/build"), exist_ok=True)
    #         os.makedirs(op.join(HERE, "docs/source"), exist_ok=True)
    #     if testing:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_testing}"
    #         )
    #     if formatting:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_formatting}"
    #         )
    #     if jupyter:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_jupyter}"
    #         )

    #         extensions_file = op.abspath(
    #             op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
    #         )
    #         with open(extensions_file) as fp:
    #             extensions = yaml.safe_load(fp)

    #         for extension in extensions["extensions"]:
    #             extn_name = "@{channel}/{name}@{version}".format(**extension)
    #             c.run(f"jupyter labextension install --no-build {extn_name}",)

    #         out = c.run("jupyter lab build")
    #     if extras:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_extras}"
    #         )
    #         os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)
    #     if ts:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_ts}"
    #         )
    #     if pyspark:
    #         c.run(
    #             f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_pyspark}"
    #         )

@task(name="setup_addon")
def setup_addon(
    c,
    platform=PLATFORM,
    env=DEV_ENV,
    all=False,
    documentation=False,
    testing=False,
    formatting=False,
    jupyter=False,
    extras=False,
    ts=False,
    tareg=False,
):
    """Installs add on packages related to documentation, testing or code-formatting.

    Dependencies related to documentation, testing or code-formatting can be installed on demand.
    By Specifying `--documentation`, documentation related packages get installed. Similarly to install testing
    or formatting related packages, flags `--testing` `formatting` will do the needful installations.
    """
    env_name = _get_env_name(env)

    _setup_addon_common(c,
    env_name,
    platform=platform,
    env=env,
    all=all,
    documentation=documentation,
    testing=testing,
    formatting=formatting,
    jupyter=jupyter,
    extras=extras,
    ts=ts,
    tareg=tareg,)


@task(name="setup_addon_pyspark")
def setup_addon_pyspark(
    c,
    platform=PLATFORM,
    env=DEV_ENV,
    all=False,
    documentation=False,
    testing=False,
    formatting=False,
    jupyter=False,
    extras=False,
    ts=False,
    tareg=False,
):
    """Installs add on packages related to documentation, testing or code-formatting for pyspark envs.

    Dependencies related to documentation, testing or code-formatting can be installed on demand.
    By Specifying `--documentation`, documentation related packages get installed. Similarly to install testing
    or formatting related packages, flags `--testing` `formatting` will do the needful installations.
    """
    env_name = _get_env_name_pyspark(env)

    _setup_addon_common(c,
    env_name,
    platform=platform,
    env=env,
    all=all,
    documentation=documentation,
    testing=testing,
    formatting=formatting,
    jupyter=jupyter,
    extras=extras,
    ts=ts,
    tareg=tareg,)

@task(name="format-code")
def format_code(c, platform=PLATFORM, env=DEV_ENV, path="."):
    env_name = _get_env_name(env)
    with py_env(c, env_name):
        c.run(f"""black "{path}" """, warn=True)
        c.run(f"""isort -rc "{path}" """)


@task(name="refresh-version")
def refresh_version(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(env)
    with py_env(c, env_name):
        res = c.run(f"""python "{HERE}"/setup.py --version""")
    return res.stdout


@task(
    name="run-notebooks",
    help={
        "template": (
            "Specifies the template folder to run"
            "``{regression|classification|all}``"
            "If all then all the templates are run."
        ),
        "notebookid": "Specifies id of notebooks. Must be one of ``{01|02|03|all}``",
        "timeout": "Maximum execution time beyond which the notebook returns an exception. default is 600 secs.",
        "refresh": "If `True`, notebooks are run and if successful refresh is also done else, notebooks are run for errors.",
    },
)
def run_notebook(
    c,
    platform=PLATFORM,
    env=DEV_ENV,
    template="all",
    notebookid="all",
    timeout=600,
    refresh=False,
):
    """Run notebooks to checks for any errors."""
    env_name = _get_env_name(env)
    with py_env(c, env_name):
        res = c.run(
            f"""python "{NOTEBOOK_FOLDER}"/utils.py {template} {notebookid} {timeout} {refresh}"""
        )
    return res.stdout


@task(name="info")
def setup_info(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(env)
    with py_env(c, env_name):
        res = c.run(f"pip list")
    return res.stdout


@task(name="build-docker")
def _build_docker_image(c):
    with tempfile.TemporaryDirectory() as tempdir:
        docker_file = op.join(HERE, "deploy", "docker", "Dockerfile")
        shutil.copyfile(docker_file, op.join(tempdir, "Dockerfile"))
        docker_file = op.join(tempdir, "Dockerfile")
        template = op.basename(HERE)
        if template == "regression-py":
            tag = "ct-reg-py"
        elif template == "classification-py":
            tag = "ct-class-py"
        elif template == "tpo-py":
            tag = "ct-tpo-py"
        elif template == "rtm-py":
            tag = "ct-rtm-py"
        elif template == "ebo-py":
            tag = "ct-ebo-py"
        elif template == "reco-py":
            tag = "ct-reco-py"
        else:
            raise ValueError(f"Unknown template : {template}")
        shutil.copytree(op.join(HERE, "deploy"), op.join(tempdir, "deploy"))
        shutil.copytree(op.join(HERE, "production"), op.join(tempdir, "production"))
        shutil.copytree(
            op.join(HERE, "src", "ta_lib"), op.join(tempdir, "src", "ta_lib")
        )
        shutil.copyfile(op.join(HERE, "setup.py"), op.join(tempdir, "setup.py"))
        shutil.copyfile(op.join(HERE, "setup.cfg"), op.join(tempdir, "setup.cfg"))
        build_context = tempdir
        c.run(f"""docker build -f "{docker_file}" -t {tag} "{build_context}" """)


@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``ct-full-{windows|linux}-{cpu|gpu}-{64|32}``"
        ),
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
    }
)
def setup_ci_env(c, platform=PLATFORM, force=False):
    """Setup a new CI environment for git.

    Creates a new conda environment with the dependencies specified in the file
    ``env/{platform}-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """
    env = "ci"
    req_file = ""
    force_flag = "" if not force else "--force"
    env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-full-{platform}-{env}.yml"))
    req_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, "ext-tiger-libs-req.txt")
    )

    req_flag = True
    if not op.isfile(env_file):
        raise ValueError(f"""The conda env file is not found : "{env_file}" """)
    if not op.isfile(req_file):
        req_flag = False
    env_name = _get_env_name(env)

    out = c.run(f"""conda env create --name {env_name} --file "{env_file}"  {force_flag}""")

    # install the code-template modules
    with py_env(c, env_name):

        # install pip requirements
        if req_flag:
            c.run(f"""python -m pip install -r "{req_file}" --no-deps""")

        # install the current package
        c.run(f"""python -m pip install -e "{HERE}" """)


_create_task_collection(
    "dev",
    setup_env,
    setup_env_pyspark,
    format_code,
    refresh_version,
    run_notebook,
    setup_addon,
    setup_addon_pyspark,
    setup_info,
    _build_docker_image,
    setup_ci_env,
)


# -------------
# Test/QC tasks
# --------------
@task(name="qc")
def run_qc_test(c, platform=PLATFORM, env=DEV_ENV, fail=False):
    env_name = _get_env_name(env)
    with py_env(c, env_name):
        # This runs flake8, flake8-black, flake8-isort, flake8-bandit,
        # flake8-docstring
        c.run(f"""python -m flake8 "{SOURCE_FOLDER}" """, warn=(not fail))


@task(name="unittest")
def run_unit_tests(c, platform=PLATFORM, env=DEV_ENV, markers=None):
    env_name = _get_env_name(env)
    markers = "" if markers is None else f"-m {markers}"
    with py_env(c, env_name):
        for usecase in TESTS_USECASE:
            # FIXME: Add others, flake9-black, etc
            test_usecase = op.join(TESTS_FOLDER, usecase)
            c.run(f"""pytest -v "{test_usecase}" {markers}""")


@task(name="vuln")
def run_vulnerability_test(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(env)
    # FIXME: platform agnostic solution: get the output from conda and then merge in python
    with py_env(c, env_name):
        c.run(
            f'conda list | tail -n +4 | tr -s " " " " '
            '| cut -f 1,2 -d " " | sed "s/\ /==/g" '
            "| safety check --stdin",
        )


@task(name="all")
def run_all_tests(c):
    run_qc_test(c)
    run_vulnerability_test(c)
    run_unit_tests(c)
    validate_env(c)

def _get_expected_env_list(env_files):
    expected_list = []
    expected_list.append("ta-lib=={}".format(get_package_version(SOURCE_FOLDER)))
    
    # collecting all the dependencies from the env files
    for env_file in env_files:
        if env_file.endswith('.yaml') or env_file.endswith('.yml'):
            # Process YAML files
            with open(env_file) as fp:
                env_cfg = yaml.safe_load(fp)
                if 'dependencies' in env_cfg:
                    for i in env_cfg["dependencies"]:
                        if type(i) is not dict:
                            expected_list.append(i)
                        else:
                            expected_list += i.get("pip", [])
        elif env_file.endswith('.txt'):
            # Process TXT files (assuming each line is a package specifier)
            with open(env_file) as fp:
                lines = fp.readlines()
                expected_list += [line.strip().split('#', 1)[0].strip() for line in lines if not line.strip().startswith('#')]

    def clean_package_name(s):
        if "git+" in s:
            s = s.split("/")[-1].split(".git")
            s = s[0] + "==" + s[-1].split("v")[-1]
        return s.replace("_", "-").lower()

    expected_list = [
        clean_package_name(i)
        for i in expected_list
        if not ("nodejs" in i or "mlflow" in i or "tigerml" in i)
    ]

    return expected_list

def _get_installed_list(c, env_name):
    # with py_env(c, env_name):
    #     import pkg_resources

    #     installed_packages = pkg_resources.working_set
    #     installed_packages_list = sorted(
    #         ["%s==%s" % (i.key.lower(), i.version) for i in installed_packages]
    #     )
    installed_packages_list = None
    with py_env(c, env_name):
        installed_packages = c.run("conda list", hide='both')
    installed_packages_str = str(installed_packages)
    installed_packages_info = installed_packages_str[installed_packages_str.rfind("#"):
        installed_packages_str.rfind("(no stderr)")].strip().split('\n')
    installed_packages_list = []
    for package_info in installed_packages_info:
        package_info_list = package_info.split()
        installed_packages_list.append(f'{package_info_list[0]}=={package_info_list[1]}')

    return installed_packages_list

@task(name="val-env")
def validate_env(
    c,
    platform=PLATFORM,
    env=DEV_ENV,
    all=False,
    documentation=False,
    testing=False,
    formatting=False,
    jupyter=False,
    extras=False,
    ts=False,
    tareg=False,
    usecase=None
    ):

    env_name = _get_env_name(env)

    # for core packages
    env_files = [op.join(PIP_REQ_FOLDER, f"ct-core-{env}.txt")]

    # addon files check
    addon_list = []
    if documentation or all:
        addon_list.append(addon_dict["documentation"])
    if testing or all:
        addon_list.append(addon_dict["testing"])
    if formatting or all:
        addon_list.append(addon_dict["formatting"])
    if jupyter or all:
        addon_list.append(addon_dict["jupyter"])
    if extras or all:
        addon_list.append(addon_dict["extras"])
    if ts or all:
        addon_list.append(addon_dict["ts"])
    if tareg or all:
        addon_list.append(addon_dict["tareg"])

    addon_file_list = _addon_file_paths(platform=platform, env=env, addon_list=addon_list)

    env_files.extend(addon_file_list)

    if usecase:
        if usecase in ["tpo", "mmx", "ebo", "rtm"]:
            usecase_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"ct-{usecase}-{env}.yml"))
            env_files.append(usecase_file)
        else:
            raise FileNotFoundError(
                "This is not a valid usecase. Valid usecases -> tpo or mmx or ebo")

    # expected and installed packages list
    expected_list = _get_expected_env_list(env_files)
    installed_list = _get_installed_list(c, env_name)

    # working on name discrepancies in packages
    installed_dict = dict(zip([x.replace("_", "-") for x in installed_list], installed_list))
    expected_dict = dict(zip([x.replace("_", "-") for x in expected_list], expected_list))

    # actual validation
    package_diff_keys = list(set(expected_dict.keys()) - set(installed_dict.keys()))
    package_diff = [v for (k, v) in expected_dict.items() if k in package_diff_keys]

    # NOTE: these package won't cause issues anymore but still keeping them to
    # keep the code because in future their might be some packages that we need
    # to ignore the versions of.
    final_package_diff = []

    ignore_list_no_version = ['pip', 'pyyaml', 'invoke']
    for pack in package_diff:
        for p in ignore_list_no_version:
            if p in pack:
                pass
            else:
                final_package_diff.append(pack)

    final_package_diff = list(set(final_package_diff))

    if len(final_package_diff) > 0:
        print(f"Following packages are missing or have wrong versions {final_package_diff}")
    else:
        print("You are all good!")

    # env_files = [
    #     op.join(CONDA_ENV_FOLDER, filename)
    #     for filename in os.listdir(CONDA_ENV_FOLDER)
    #     if fnmatch.fnmatch(filename, 'addon-*.yml')
    #     # for env_file in os.listdir(CONDA_ENV_FOLDER)
    #     # if (".yml" in env_file) # & (PLATFORM in env_file)
    # ]
    # env_files.append(op.join(CONDA_ENV_FOLDER, f"{platform}-{env}.yml"))
    # expected_list = ["ta-lib==1.1.0"]

    # for env_file in env_files:
    #     with open(env_file) as fp:
    #         env_cfg = yaml.safe_load(fp)
    #         # print(env_cfg)
    #     if 'dependencies' in env_cfg:
    #         for i in env_cfg["dependencies"]:
    #             if type(i) is not dict:
    #                 expected_list.append(i)
    #             else:
    #                 expected_list = expected_list + i["pip"]

    # def clean_package_name(s):
    #     if "git+" in s:
    #         s = s.split("/")[-1].split(".git")
    #         s = s[0] + "==" + s[-1].split("v")[-1]
    #     return s.replace("_", "-").lower()

    # expected_list = [
    #     clean_package_name(i)
    #     for i in expected_list
    #     if not ("nodejs" in i or "mlflow" in i or "tigerml" in i)
    # ]

    # with py_env(c, env_name):
    #     import pkg_resources

    #     installed_packages = pkg_resources.working_set
    #     installed_packages_list = sorted(
    #         ["%s==%s" % (i.key.lower(), i.version) for i in installed_packages]
    #     )




_create_task_collection(
    "test",
    run_qc_test,
    run_vulnerability_test,
    run_unit_tests,
    run_all_tests,
    validate_env,
)


# -----------
# Build tasks
# -----------
@task(name="docs")
def build_docs(c, platform=PLATFORM, env=DEV_ENV, regen_api=True, update_credits=False):
    env_name = _get_env_name(env)
    with py_env(c, env_name):
        if regen_api:
            code_path = op.join(HERE, "docs", "source", "_autosummary")
            if os.path.exists(code_path):
                _clean_rmtree(code_path)
            os.makedirs(code_path, exist_ok=True)

        # FIXME: Add others, flake9-black, etc
        if update_credits:
            credits_path = op.join(HERE, "docs", "source", "_credits")
            if os.path.exists(credits_path):
                _clean_rmtree(credits_path)
            os.makedirs(credits_path, exist_ok=True)
            authors_path = op.join(HERE, "docs")
            token = os.environ["GITHUB_OAUTH_TOKEN"]
            c.run(f"""python "{authors_path}"/generate_authors_table.py {token} {token}""")
        c.run(
            "cd docs/source && sphinx-build -T -E -W --keep-going -b html -d ../build/doctrees  . ../build/html"
        )


_create_task_collection("build", build_docs)


# -----------
# Launch stuff
# -----------
@task(name="jupyterlab")
def start_jupyterlab(
    c, platform=PLATFORM, env=DEV_ENV, ip="localhost", port=8080, token="", password=""
):
    env_name = _get_env_name(env)
    # FIXME: run as a daemon and support start/stop using pidfile
    with py_env(c, env_name):
        print(
            f"{'--'*20} \n Running jupyterlab with {env_name} environment \n {'--'*20}"
        )
        c.run(
            f"jupyter lab --ip {ip} --port {port} --NotebookApp.token={token} "
            f"--NotebookApp.password={password} --no-browser"
        )


@task(name="jupyterlab_pyspark")
def start_jupyterlab_pyspark(
    c, platform=PLATFORM, env=DEV_ENV, ip="localhost", port=8081, token="", password=""
):
    env_name = _get_env_name_pyspark(env)
    # FIXME: run as a daemon and support start/stop using pidfile
    with py_env(c, env_name):
        print(
            f"{'--'*20} \n Running jupyterlab with {env_name} environment \n {'--'*20}"
        )  # To: Delete print statement
        c.run(
            f"jupyter lab --ip {ip} --port {port} --NotebookApp.token={token} "
            f"--NotebookApp.password={password} --no-browser"
        )


@task(name="tracker-ui")
def start_tracker_ui(c, platform=PLATFORM, env=DEV_ENV, port=8082):
    env_name = _get_env_name(env)
    # FIXME: run as a daemon and support start/stop using pidfile
    # NOTE: using sqlite with more than 1 worker will cause issues
    # if scalability is needed, use a postgresql server
    with py_env(c, env_name):
        c.run(
            f"""mlflow server --port {port} --default-artifact-root "{HERE}"/mlruns """
            f"""--backend-store-uri sqlite:///"{HERE}"/mlruns/mlflow.db """
            f"--workers 1"
        )


@task(name="docs")
def start_docs_server(c, ip="127.0.0.1", port=8081):
    # FIXME: run as a daemon and support start/stop using pidfile
    print(f"Serving HTTP on {ip} port {port} (http://{ip}:{port}/)")
    c.run(
        f"python -m http.server --bind 127.0.0.1 " f"--directory docs/build/html {port}"
    )


@task(name="ipython")
def start_ipython_shell(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(env)
    # FIXME: run as a daemon and support start/stop using pidfile
    startup_script = op.join(HERE, "deploy", "ipython", "default_startup.py")
    with py_env(c, env_name):
        c.run(f"""ipython -i "{startup_script}" """)


_create_task_collection(
    "launch",
    start_jupyterlab,
    start_jupyterlab_pyspark,
    start_tracker_ui,
    start_docs_server,
    start_ipython_shell,
)


# --------------
# Root namespace
# --------------
# override any configuration for the tasks here
# FIXME: refactor defaults (constants) and set them as config after
# auto-detecting them
ns = _create_root_task_collection()
config = dict(pty=True, echo=True)

if OS == "windows":
    config["pty"] = False

ns.configure(config)
