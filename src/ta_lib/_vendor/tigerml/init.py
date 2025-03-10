# import sentry_sdk
# sentry_sdk.init("https://d54aa1f71e864d0b850fe3934a06ea61@sentry.io/1541480")
import sys
import traceback

with open("./tigerml/config.py") as fp:
    ns = {}
    exec(fp.read(), ns, ns)
__version__ = ns["__version__"]


def get_from_config(var_name):
    from . import config as C

    if hasattr(C, var_name):
        return getattr(C, var_name)
    else:
        print("No variable called {}".format(var_name))


if get_from_config("TRACKING"):
    try:
        import rollbar

        rollbar.init("006bdf068e2f4403bceb2f4b16091d24", get_from_config("ENVIRONMENT"))

        def rollbar_except_hook(exc_type, exc_value, exc_traceback):
            # Report the issue to rollbar here.
            if "tigerml" in str(traceback.extract_tb(exc_traceback)[-1]):
                rollbar.report_exc_info((exc_type, exc_value, exc_traceback))
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = rollbar_except_hook
    except:
        pass
