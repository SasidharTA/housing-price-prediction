"""The set of functions present in this module lets you set distributions on the basis of config file."""

import sys

import tensorflow_probability as tfp

from .logging_utils import get_logger

tfd = tfp.distributions
tfb = tfp.bijectors
LOG = get_logger(__name__)


def create_Normal_dist(loc_, scale_):
    """Create Normal distribution.

    Parameters
    ----------
    loc : int/float
        loc value for Normal distribution.
    scale : int/float
        scale value for Normal distribution.

    Returns
    -------
     Normal distribution.

    """
    return tfd.Normal(loc=float(loc_), scale=float(scale_))


def create_StudentT_dist(loc_, scale_):
    """Create Student_T distribution.

    Parameters
    ----------
    loc : int/float
        loc value for Student_T distribution.
    scale : int/float
        scale value for Student_T distribution..

    Returns
    -------
    Student_T distribution.

    """
    return tfd.StudentT(loc=float(loc_), scale=float(scale_), df=3)


def create_HalfCauchy_dist(loc_, scale_):
    """Create HalfCauchy distribution.

    Parameters
    ----------
    loc : int/float
        loc value for HalfCauchy distribution.
    scale : int/float
        scale value for HalfCauchy distribution.

    Returns
    -------
    HalfCauchy distribution.

    """
    return tfd.HalfCauchy(loc=float(loc_), scale=float(scale_))


def create_Cauchy_dist(loc_, scale_):
    """Create Cauchy distribution.

    Parameters
    ----------
    loc : int/float
        loc value for Cauchy distribution.
    scale : int/float
        scale value for Cauchy distribution.

    Returns
    -------
    Cauchy distribution.

    """
    return tfd.Cauchy(loc=float(loc_), scale=float(scale_))


def create_LogNormal_dist(loc_, scale_):
    """Create LogNormal distribution.

    Parameters
    ----------
    loc : int/float
        loc value for LogNormal distribution.
    scale : int/float
        loc value for LogNormal distribution.

    Returns
    -------
    LogNormal distribution.

    """
    return tfd.LogNormal(loc=float(loc_), scale=float(scale_))


def create_Gamma_dist(con_, rate_):
    """
    Create Gamma distribution.

    Parameters
    ----------
    con : int/float
        concentration value for Gamma distribution.
    rate : int/float
        rate value for Gamma distribution.

    Returns
    -------
    Gamma distribution.

    """
    return tfd.Gamma(concentration=float(con_), rate=float(rate_))


def create_Uniform_dist(low_, high_):
    """
    Create Uniform distribution.

    Parameters
    ----------
    low : int/float
        low value for Uniform distribution.
    high : int/float
        high value for Uniform distribution.

    Returns
    -------
    Uniform distribution.

    """
    return tfd.Uniform(low=int(low_), high=int(high_))


def create_Beta_dist(concentration1_, concentration0_):
    """Create Beta distribution.

    Parameters
    ----------
    concentration1 : int/float
        concentration1 value for Beta distribution.
    concentration0 : int/float
        concentration0 value for Beta distribution.

    Returns
    -------
    Beta distribution.

    """
    return tfd.Beta(
        concentration1=float(concentration1_), concentration0=float(concentration0_)
    )


def create_HalfNormal_dist(scale_):
    """Create HalfNormal distribution.

    Parameters
    ----------
    scale : int/float
        scale value for HalfNormal distribution.

    Returns
    -------
    HalfNormal distribution.

    """
    return tfd.HalfNormal(scale=float(scale_))


def select_distribution(dist_type, val1, val2):
    """To assign the distribution for Joint distribution coroutine given by user.

    Parameters
    ----------
    dist_type : string
        distribution.
    val1 : int/float
        prior loc value.
    val2 : int/float
        prior scale value.

    Raises
    ------
    ValueError
        If dist_type is not in the list of allowed distributions.

    Returns
    -------
    Selected distibution.

    """
    dist_list = [
        "Normal",
        "HalfCauchy",
        "LogNormal",
        "StudentT",
        "Gamma",
        "Cauchy",
        "Uniform",
        "Beta",
        "HalfNormal",
    ]

    if dist_type not in dist_list:
        LOG.error(
            dist_type,
            "distribution is not allowed. Please use only Normal/ HalfCauchy/ LogNormal/ Gamma/ Cauchy/ Uniform/ Beta/ HalfNormal distribution.",
        )
        sys.exit(1)
    elif dist_type == "Normal":
        return create_Normal_dist(val1, val2)
    elif dist_type == "HalfCauchy":
        return create_HalfCauchy_dist(val1, val2)
    elif dist_type == "Cauchy":
        return create_Cauchy_dist(val1, val2)
    elif dist_type == "LogNormal":
        return create_LogNormal_dist(val1, val2)
    elif dist_type == "StudentT":
        return create_StudentT_dist(val1, val2)
    elif dist_type == "Gamma":
        return create_Gamma_dist(val1, val2)
    elif dist_type == "Uniform":
        return create_Uniform_dist(val1, val2)
    elif dist_type == "Beta":
        return create_Beta_dist(val1, val2)
    elif dist_type == "HalfNormal":
        return create_HalfNormal_dist(val2)
