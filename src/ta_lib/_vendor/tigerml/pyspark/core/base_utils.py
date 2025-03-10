"""Module with utility functions that can be called at library import."""
import warnings


def silence_common_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The sklearn.metrics.classification module",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", ".*optional dependency `torch`.*")
