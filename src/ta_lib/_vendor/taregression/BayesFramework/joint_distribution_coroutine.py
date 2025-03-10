"""It creates string for joint distribution (collection of possibly interdependent distribution) and log probability density."""

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .distribution import create_HalfCauchy_dist, select_distribution
from .logging_utils import get_logger
from .utils import affine

tfd = tfp.distributions
tfb = tfp.bijectors

LOG = get_logger(__name__)


def global_intercept_param(join_dist_list, dist_param, unconstraining_bijectors):
    """
    To select distribution for global intercept.

    Parameters
    ----------
    join_dist_list : list
        list of variables name .
    dist_param : dict
        dictionary that contains information about distribution, loc and scale values.
    unconstraining_bijectors: list
        bijectors list.

    Returns
    -------
    Selected distibution for global intercept.

    """
    if "fixed_slope_global_intercept" not in join_dist_list:
        join_dist_list.append("fixed_slope_global_intercept")
        unconstraining_bijectors.append(
            dist_param["global_intercept"]["fixed_bijector"]
        )
    return select_distribution(
        dist_param["global_intercept"]["fixed_d"],
        dist_param["global_intercept"]["fixed_d_loc"],
        dist_param["global_intercept"]["fixed_d_scale"],
    )


def random_intercept_param(
    join_dist_list, dist_param, group_variable, group_count, unconstraining_bijectors
):
    """
    To select distribution for random_intercept.

    Parameters
    ----------
    join_dist_list : list
        list of variables name .
    dist_param : dict
        dictionary that contains information about distribution, loc and scale values.
    group_variable : string
        grouping column name.
    group_count : int/float
        number of unique elements in the group.
    unconstraining_bijectors: list
        bijectors list.

    Returns
    -------
    Selected distibution with given priors for random intercept.

    """
    if not all(
        item in join_dist_list
        for item in [
            "mu_intercept_" + group_variable,
            "sigma_intercept_" + group_variable,
            "intercept_" + group_variable,
        ]
    ):
        join_dist_list.extend(
            [
                "mu_intercept_" + group_variable,
                "sigma_intercept_" + group_variable,
                "intercept_" + group_variable,
            ]
        )

        unconstraining_bijectors.extend(
            [
                dist_param[group_variable]["mu_bijector"],
                dist_param[group_variable]["sigma_bijector"],
                tfb.Identity(),
            ]
        )
    return (
        select_distribution(
            dist_param[group_variable]["mu_d"],
            dist_param[group_variable]["mu_d_loc"],
            dist_param[group_variable]["mu_d_scale"],
        ),
        select_distribution(
            dist_param[group_variable]["sigma_d"],
            dist_param[group_variable]["sigma_d_loc"],
            dist_param[group_variable]["sigma_d_scale"],
        ),
    )  # sigma_intercept: hyper-prior


def random_variance_param(
    join_dist_list, dist_param, group_variable, group_count, unconstraining_bijectors
):
    """
    To select distribution for random variance.

    Parameters
    ----------
    join_dist_list : list
        list of variables name .
    dist_param : dict
        dictionary that contains information about distribution, loc and scale values.
    group_variable : string
        grouping column name.
    group_count : int/float
        number of unique elements in the group.
    unconstraining_bijectors: list
        bijectors list.

    Returns
    -------
    Selected distibution with given priors for random variance.

    """
    if not all(
        item in join_dist_list
        for item in ["sigma_intercept_" + group_variable, "intercept_" + group_variable]
    ):
        join_dist_list.extend(
            [
                "sigma_intercept_" + group_variable,
                "intercept_" + group_variable,
            ]
        )
        unconstraining_bijectors.extend(
            [
                dist_param[group_variable]["sigma_bijector"],
                tfb.Identity(),
            ]
        )

    return select_distribution(
        dist_param[group_variable]["sigma_d"],
        dist_param[group_variable]["sigma_d_loc"],
        dist_param[group_variable]["sigma_d_scale"],
    )  # sigma_intercept: hyper-prior


def fixed_slope_param(join_dist_list, dist_param, variable, unconstraining_bijectors):
    """
    To select distribution for fixed slope.

    Parameters
    ----------
    join_dist_list : list
        list of variables name .
    dist_param : dict
        dictionary that contains information about distribution, loc and scale values.
    variable : string
        column name of idv.
    unconstraining_bijectors: list
        bijectors list.

    Returns
    -------
    Selected distibution with given priors for fixed slope variable.

    """
    s = "fixed_slope_" + variable
    if s not in join_dist_list:
        join_dist_list.extend(["fixed_slope_" + variable])
        unconstraining_bijectors.append(dist_param[variable]["fixed_bijector"])
    return select_distribution(
        dist_param[variable]["fixed_d"],
        dist_param[variable]["fixed_d_loc"],
        dist_param[variable]["fixed_d_scale"],
    )


def random_slope_param(
    join_dist_list,
    dist_param,
    variable,
    group_variable,
    group_count,
    unconstraining_bijectors,
):
    """
    To select distribution for random slope.

    Parameters
    ----------
    join_dist_list : list
        list of variable names .
    dist_param : dict
        dictionary that contains information about distribution, loc and scale values.
    variable : string
        column name of idv.
    group_variable : string
        grouping column name.
    group_count : int/float
        number of unique elements in the group.
    unconstraining_bijectors: list
        bijectors list.

    Returns
    -------
    Selected distibution with given priors for random slope.

    """
    if not all(
        item in join_dist_list
        for item in [
            "mu_slope_" + variable + "_" + group_variable,
            "sigma_slope_" + variable + "_" + group_variable,
            "slope_" + variable + "_" + group_variable,
        ]
    ):
        join_dist_list.extend(
            [
                "mu_slope_" + variable + "_" + group_variable,
                "sigma_slope_" + variable + "_" + group_variable,
                "slope_" + variable + "_" + group_variable,
            ]
        )
        unconstraining_bijectors.extend(
            [
                dist_param[variable]["mu_bijector"],
                dist_param[variable]["sigma_bijector"],
                tfb.Identity(),
            ]
        )
    return (
        select_distribution(
            dist_param[variable]["mu_d"],
            dist_param[variable]["mu_d_loc"],
            dist_param[variable]["mu_d_scale"],
        ),
        select_distribution(
            dist_param[variable]["sigma_d"],
            dist_param[variable]["sigma_d_loc"],
            dist_param[group_variable]["sigma_d_scale"],
        ),
    )  # sigma_slope: hyper-prior


def error(join_dist_list, dist_param, unconstraining_bijectors):
    """
    To add the error term for target variable.

    Parameters
    ----------
    join_dist_list : list
        list of variable names .
    dist_param : dict
        dictionary that contains information about distribution, loc and scale values.
    unconstraining_bijectors: list
        bijectors list.

    Returns
    -------
    Error term to the joint distribution coroutine.

    """
    if "sigma_target" not in join_dist_list:
        join_dist_list.append("sigma_target")
        (
            unconstraining_bijectors.append(
                dist_param["sigma_target"]["fixed_bijector"]
            )
            if "sigma_target" in dist_param
            else unconstraining_bijectors.append(tfb.Exp())
        )

    if "sigma_target" in dist_param:
        distribution = select_distribution(
            dist_param["sigma_target"]["fixed_d"],
            dist_param["sigma_target"]["fixed_d_loc"],
            dist_param["sigma_target"]["fixed_d_scale"],
        )

    else:
        distribution = create_HalfCauchy_dist(0, 5)

    return distribution


def create_joint_dist_co(  # noqa: C901
    model_config_df,
    data_df,
    join_dist_list,
    dist_param,
    unconstraining_bijectors,
    tensor_d,
):
    """
    To apply joint distribution Coroutine over component distributions.

    Parameters
    ----------
    model_config_df : dataframe
        dataframe with dv, idvs and distribution information.
    data_df : dataframe
        data.
    join_dist_list : list
        list of variable names.
    dist_param : dictionary
        dictionary that contains information about distribution, loc and scale values.
    unconstraining_bijectors: list
        bijectors list.
    tensor_d: dictionary
        data converted in tensor


    Returns
    -------
    Joint distribution coroutine parameterized by a distribution-making generator.

    """
    Root = tfd.JointDistributionCoroutine.Root
    dist_dict = {}

    def model():
        try:
            fixed_effect = 0
            random_effect = 0
            for _, row in model_config_df.iterrows():
                if "intercept" in row["IDV"]:
                    if row["RandomEffect"] == 1:
                        # add random intercept term
                        random_factor = row["RandomFactor"]
                        group_levels = tf.cast(
                            data_df[random_factor].nunique(), tf.int32
                        )
                        if (row["mu_d"] == "Gamma") or (row["sigma_d"] == "Gamma"):
                            d1, d2 = random_intercept_param(
                                join_dist_list,
                                dist_param,
                                random_factor,
                                group_levels,
                                unconstraining_bijectors,
                            )
                            dist_dict["mu_intercept_" + random_factor] = yield Root(d1)
                            dist_dict["sigma_intercept_" + random_factor] = yield Root(
                                d2
                            )

                            loc_ = affine(
                                tf.ones([group_levels]),
                                dist_dict["mu_intercept_" + random_factor][
                                    ..., tf.newaxis
                                ],
                            )

                            scale_ = tf.cast(
                                tf.transpose(
                                    group_levels
                                    * [dist_dict["sigma_intercept_" + random_factor]]
                                ),
                                dtype=tf.float32,
                            )

                            dist_dict["intercept_" + random_factor] = (
                                yield tfd.Independent(
                                    tfd.Gamma(concentration=loc_, rate=scale_),
                                    reinterpreted_batch_ndims=1,
                                )
                            )

                        elif pd.isna(row["mu_d"]):
                            d1 = random_variance_param(
                                join_dist_list,
                                dist_param,
                                random_factor,
                                group_levels,
                                unconstraining_bijectors,
                            )
                            dist_dict["sigma_intercept_" + random_factor] = yield Root(
                                d1
                            )
                            dist_dict["intercept_" + random_factor] = (
                                yield tfd.MultivariateNormalDiag(
                                    loc=tf.zeros([group_levels]),
                                    scale_identity_multiplier=dist_dict[
                                        "sigma_intercept_" + random_factor
                                    ],
                                )
                            )

                        else:
                            d1, d2 = random_intercept_param(
                                join_dist_list,
                                dist_param,
                                random_factor,
                                group_levels,
                                unconstraining_bijectors,
                            )
                            dist_dict["mu_intercept_" + random_factor] = yield Root(d1)
                            dist_dict["sigma_intercept_" + random_factor] = yield Root(
                                d2
                            )

                            loc_ = affine(
                                tf.ones([group_levels]),
                                dist_dict["mu_intercept_" + random_factor][
                                    ..., tf.newaxis
                                ],
                            )
                            dist_dict["intercept_" + random_factor] = (
                                yield tfd.MultivariateNormalDiag(
                                    loc=loc_,
                                    scale_identity_multiplier=dist_dict[
                                        "sigma_intercept_" + random_factor
                                    ],
                                )
                            )

                        # likelihood loc term
                        random_effect += tf.gather(
                            dist_dict["intercept_" + random_factor],
                            tensor_d[random_factor],
                            axis=-1,
                        )

                    else:
                        # add global/fixed intercept term
                        d1 = global_intercept_param(
                            join_dist_list, dist_param, unconstraining_bijectors
                        )
                        dist_dict["fixed_slope_global_intercept"] = yield Root(d1)

                        fixed_effect = fixed_effect + affine(
                            tensor_d[row["IDV"]],
                            dist_dict["fixed_slope_global_intercept"][..., tf.newaxis],
                        )

                else:
                    if row["RandomEffect"] == 1:
                        # add random slope term
                        random_factor = row["RandomFactor"]
                        group_levels = tf.cast(
                            data_df[random_factor].nunique(), tf.int32
                        )
                        variable = row["IDV"]
                        if (row["mu_d"] == "Gamma") or (row["sigma_d"] == "Gamma"):
                            d1, d2 = random_slope_param(
                                join_dist_list,
                                dist_param,
                                row["IDV"],
                                random_factor,
                                group_levels,
                                unconstraining_bijectors,
                            )
                            dist_dict["mu_slope_" + variable + "_" + random_factor] = (
                                yield Root(d1)
                            )
                            dist_dict[
                                "sigma_slope_" + variable + "_" + random_factor
                            ] = yield Root(d2)

                            loc_ = affine(
                                tf.ones([group_levels]),
                                dist_dict["mu_slope_" + variable + "_" + random_factor][
                                    ..., tf.newaxis
                                ],
                            )

                            scale_ = tf.cast(
                                tf.transpose(
                                    group_levels
                                    * [
                                        dist_dict[
                                            "sigma_slope_"
                                            + variable
                                            + "_"
                                            + random_factor
                                        ]
                                    ]
                                ),
                                dtype=tf.float32,
                            )

                            dist_dict["slope_" + variable + "_" + random_factor] = (
                                yield tfd.Independent(
                                    tfd.Gamma(concentration=loc_, rate=scale_),
                                    reinterpreted_batch_ndims=1,
                                )
                            )

                        else:
                            d1, d2 = random_slope_param(
                                join_dist_list,
                                dist_param,
                                row["IDV"],
                                random_factor,
                                group_levels,
                                unconstraining_bijectors,
                            )

                            dist_dict["mu_slope_" + variable + "_" + random_factor] = (
                                yield Root(d1)
                            )
                            dist_dict[
                                "sigma_slope_" + variable + "_" + random_factor
                            ] = yield Root(d2)

                            loc_ = affine(
                                tf.ones([group_levels]),
                                dist_dict["mu_slope_" + variable + "_" + random_factor][
                                    ..., tf.newaxis
                                ],
                            )
                            dist_dict["slope_" + variable + "_" + random_factor] = (
                                yield tfd.MultivariateNormalDiag(
                                    loc=loc_,
                                    scale_identity_multiplier=dist_dict[
                                        "sigma_slope_" + variable + "_" + random_factor
                                    ],
                                )
                            )

                        # likelihood loc term
                        random_effect = random_effect + affine(
                            tensor_d[row["IDV"]],
                            tf.gather(
                                dist_dict["slope_" + variable + "_" + random_factor],
                                tensor_d[random_factor],
                                axis=-1,
                            ),
                        )

                    else:
                        # add fixed slope term
                        if row["IDV"] != "sigma_target":
                            variable = row["IDV"]
                            d1 = fixed_slope_param(
                                join_dist_list,
                                dist_param,
                                row["IDV"],
                                unconstraining_bijectors,
                            )
                            dist_dict["fixed_slope_" + variable] = yield Root(d1)

                            fixed_effect = fixed_effect + affine(
                                tensor_d[row["IDV"]],
                                dist_dict["fixed_slope_" + variable][..., tf.newaxis],
                            )

            # insert error term
            dist_dict["sigma_target"] = yield Root(
                error(join_dist_list, dist_param, unconstraining_bijectors)
            )

            linear_response = fixed_effect + random_effect
            if model_config_df["DV"].iloc[0] not in join_dist_list:
                join_dist_list.append(model_config_df["DV"].iloc[0])

            yield tfd.MultivariateNormalDiag(
                loc=linear_response, scale_identity_multiplier=dist_dict["sigma_target"]
            )

        except Exception as e:
            LOG.error("Error while creating joint distribution sequential")
            LOG.error(e)

    return tfd.JointDistributionCoroutine(model)
