"""It contails all routine functions."""

import math
import sys
import warnings

import numpy as np
import tensorflow as tf

from .logging_utils import get_logger

LOG = get_logger(__name__)

warnings.filterwarnings("ignore")

thismodule = sys.modules[__name__]


def get_tensor_var(variable, data_df):
    """To Convert data column to a Tensor.

    Parameters
    ----------
        variable: str
            column name.
        data_df: pd.DataFrame

    Raises
    ------
        TypeError: If no conversion function is registered for value to dtype.
        RuntimeError: If a registered conversion function returns an invalid value.
        ValueError: If the value is a tensor not of given dtype in graph mode.

    Returns
    -------
        tf.float32
        Converted data column to Tensor float 32 data column.
    """
    return tf.convert_to_tensor(data_df[variable], dtype=tf.float32)


def get_tensor_cat(variable, data_df):
    """To Convert Category column to a Tensor.

    Parameters
    ----------
        variable: str
            column name.
        data_df: pd.DataFrame

    Raises
    ------
        TypeError: If no conversion function is registered for value to dtype.
        RuntimeError: If a registered conversion function returns an invalid value.
        ValueError: If the value is a tensor not of given dtype in graph mode.

    Returns
    -------
        tf.float32
        Converted category column to Tensor float 32 data column.
    """
    return tf.convert_to_tensor(data_df[variable])


def tensor_to_dictionary(join_dist_list, SamplesTuple, mapped_df, spec_dict):
    """To Save tensors of each level in group into dictionary.

    Returns
    -------
        dict
    """
    dict_sample = {}
    var_list = [
        s
        for s in join_dist_list[:-1]
        if ((("mu_" not in s) and ("sigma_" not in s)) or ("sigma_target" in s))
    ]

    for var in var_list:
        value = getattr(SamplesTuple, var).numpy()
        if value.ndim > 2:
            # if "intercept_" in var:
            #     var_n = var.split("_", 1)[1]
            # elif "slope_" in var:
            #     var_n = var.split("_", 2)[2]
            var_n = [s for s in spec_dict["group_cols"] if s in var][0]
            for i in range(value.shape[2]):
                n_value = value[:, :, i]
                j = mapped_df[var_n + "_original"][mapped_df[var_n] == i].unique()
                dict_sample[var + str(j)] = np.swapaxes(
                    n_value, 1, 0
                )  # for changing chain and draw axis
        else:
            dict_sample[var] = np.swapaxes(value, 1, 0)
    return dict_sample


def group_estimates(approx_param, mapped_df, spec_dict):
    """To Save point estimates of each level in group into dictionary.

    Returns
    -------
        dict
    """
    dict_estimates = {}

    for k, v in approx_param.items():
        if (("mu_" not in k) and ("sigma_" not in k)) or ("sigma_target" in k):

            if (
                ("global_intercept" in k)
                or ("fixed_slope_" in k)
                or ("sigma_target" in k)
            ):
                dict_estimates[k] = {"mu": v["mu"], "sd": v["sd"]}

            else:
                var_n = [s for s in spec_dict["group_cols"] if s in k][0]
                for i in range(len(v["sd"])):
                    j = mapped_df[var_n + "_original"][mapped_df[var_n] == i].unique()
                    dict_estimates[k + str(j)] = {"mu": v["mu"][i], "sd": v["sd"][i]}

    return dict_estimates
    # var_list = [
    #     s for s in join_dist_list[:-1] if ( (("mu_" not in s) and ("sigma_" not in s)) or ("sigma_target" in s) )
    # ]

    # for var in var_list:
    #     value = getattr(SamplesTuple, var).numpy()
    #     if value.ndim > 2:
    #         # if "intercept_" in var:
    #         #     var_n = var.split("_", 1)[1]
    #         # elif "slope_" in var:
    #         #     var_n = var.split("_", 2)[2]
    #         var_n=[s for s in spec_dict["group_cols"] if s in var][0]
    #         for i in range(value.shape[2]):
    #             n_value = value[:, :, i]
    #             j = mapped_df[var_n + "_original"][mapped_df[var_n] == i].unique()
    #             dict_sample[var + str(j)] = np.swapaxes(
    #                 n_value, 1, 0
    #             )  # for changing chain and draw axis
    #     else:
    #         dict_sample[var] = np.swapaxes(value, 1, 0)
    # return dict_sample


# @tf.function
def affine(x, kernel_diag, bias=tf.zeros([])):
    """To make kernel_diag * x + bias` with broadcasting.

    Parameters
    ----------
        x: tensor
            tensor for conversion
        kernel_diag: tensor
            tensor for kernel
        bias: tensor
            tf method to generate bias( delfault tf.zeros([]))

    Raises
    ------
        TypeError: if datatype is not a tensor

    Returns
    -------
        tensor
        kernel_diag * x + bias.
    """
    kernel_diag = tf.ones_like(x) * kernel_diag
    bias = tf.ones_like(x) * bias
    return x * kernel_diag + bias


def check_values(initial_state):
    """
    Check initial states.

    Parameters
    ----------
    initial_state : tensor
        initial state of variables.

    Returns
    -------
    None.

    """
    LOG.info("Checking initial states.")

    for i in range(len(initial_state)):
        # LOG.error(tf.debugging.check_numerics(initial_state[i], msg, name=None))
        value = initial_state[i].numpy()
        if np.isinf(value).any():
            LOG.warning("Initial state for any fixed/random effect variable is inf.")
        elif np.isnan(value).any():
            LOG.warning("initial state for any fixed/random effect variable is nan.")


def check_samples(dict_sample):
    """
    Check samples.

    Parameters
    ----------
    dict_sample : dictionary
        posterior samples of variables.

    Returns
    -------
    None.

    """
    LOG.info("Checking samples.")

    for key, value in dict_sample.items():
        l = []  # noqa: E741
        for i in range(len(value)):
            if (math.inf in value[i]) or (np.isnan(value[i]).any()):
                LOG.error(
                    "Samples contain inf or nan as value. Please provide better priors."
                )
                sys.exit(1)

            elif len(set(value[i])) == 1:
                l.append(i)
        if l:
            LOG.error(
                "All the samples of "
                + key
                + " for chain no "
                + str(l)
                + " are either same or zero. Bayesian posterior will fail to converge. Please provide better priors."
            )
            # sys.exit(1)


def reduce_samples(var_samples, reduce_fn):
    """To Reduce Samples across leading two dims using reduce_fn.

    Parameters
    ----------
        var_samples: list
            list of samples
        reduce_fn: func
            mean for samples

    Raises
    ------
        Error while reducing sample

    Returns
    -------
        ndarray
    """
    try:
        if isinstance(var_samples, tf.Tensor):
            var_samples = var_samples.numpy()
        var_samples = np.reshape(var_samples, (-1,) + var_samples.shape[2:])

    except Exception as e:
        LOG.warning("----Error while reducing sample----")
        LOG.exception(e)

    return np.apply_along_axis(reduce_fn, axis=0, arr=var_samples)


def sample_mean(samples):
    """To generate a mean of the samples.

    Parameters
    ----------
        samples: list
            list of samples

    Raises
    ------
        Error while reducing sample

    Returns
    -------
        list(tensor)
        reduced samples
    """
    return reduce_samples(samples, np.mean)
