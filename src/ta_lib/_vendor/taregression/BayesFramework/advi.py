import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.transformed_kernel import (
    make_transformed_log_prob,
)

from .logging_utils import get_logger

tfd = tfp.distributions
tfb = tfp.bijectors

LOG = get_logger(__name__)


def transform_log_prob(bijectors, joint_dist_model_log_prob):
    """
    Wrap logp so that all parameters are in the Real domain.

    Parameters
    ----------
    bijectors : list
        list of bijectors.
    joint_dist_model_log_prob : function
        log prob function.

    Returns
    -------
    None.
    """
    contextual_effect_posterior = make_transformed_log_prob(
        joint_dist_model_log_prob,
        bijectors,
        direction="forward",
        # TODO(b/72831017): Disable caching until gradient linkage
        # generally works.
        enable_bijector_caching=False,
    )

    return contextual_effect_posterior


def build_meanfield_advi(sample_t, join_dist_list, observed_node=-1):
    """The inputted jointdistribution needs to be a batch version."""
    # Sample to get a list of Tensors
    list_of_values = sample_t  # <== sample([]) might not work
    # Remove the observed node
    list_of_values.pop(observed_node)

    # Iterate the list of Tensor to a build a list of Normal distribution (i.e.,the Variational posterior)
    distlist = []
    for i, value in enumerate(list_of_values):
        dtype = value.dtype
        rv_shape = value[0].shape
        loc = tf.Variable(
            tf.random.normal(rv_shape, dtype=dtype),
            name="meanfield_%s_mu" % i,
            dtype=dtype,
        )
        scale = tfp.util.TransformedVariable(
            tf.fill(rv_shape, value=tf.constant(0.02, dtype)),
            tfb.Softplus(),
            name="meanfield_%s_scale" % i,
        )

        approx_node = tfd.Normal(loc=loc, scale=scale)
        if loc.shape == ():
            distlist.append(approx_node)
        else:
            distlist.append(
                # TODO: make the reinterpreted_batch_ndims more flexible (for minibatch etc)
                tfd.Independent(approx_node, reinterpreted_batch_ndims=1)
            )
    LOG.info(distlist)
    # pass list to JointDistribution to initiate the meanfield advi
    meanfield_advi = tfd.JointDistributionSequential(distlist)
    return meanfield_advi


@tf.function  # (experimental_compile=True)
def run_approximation(
    contextual_effect_posterior, advi, opt, num_variational_steps, seed
):
    loss_ = tfp.vi.fit_surrogate_posterior(
        contextual_effect_posterior,
        surrogate_posterior=advi,
        optimizer=opt,
        sample_size=10,
        num_steps=num_variational_steps,
        seed=seed,
    )
    return loss_


def get_bijector(model_config_df, dist_param):

    dist = ["HalfCauchy", "LogNormal", "Gamma", "HalfNormal"]
    try:
        bijectors = []
        for _, row in model_config_df.iterrows():
            if row["RandomEffect"] == 1:
                if "intercept" in row["IDV"]:
                    variable = row["RandomFactor"]
                else:
                    variable = row["IDV"]
                if (dist_param[variable]["mu_d"] in dist) and (
                    dist_param[variable]["sigma_d"] in dist
                ):
                    bijectors.extend(
                        [
                            tfb.Exp(),
                            tfb.Exp(),
                            # tfb.Identity()
                        ]
                    )
                elif (dist_param[variable]["mu_d"] in dist) and (
                    dist_param[variable]["sigma_d"] not in dist
                ):
                    bijectors.extend(
                        [
                            tfb.Exp(),
                            dist_param[variable]["sigma_bijector"],
                            # tfb.Identity()
                        ]
                    )
                elif (dist_param[variable]["mu_d"] not in dist) and (
                    dist_param[variable]["sigma_d"] in dist
                ):
                    bijectors.extend(
                        [
                            dist_param[variable]["mu_bijector"],
                            tfb.Exp(),
                            # tfb.Identity()
                        ]
                    )

                else:
                    bijectors.extend(
                        [
                            dist_param[variable]["mu_bijector"],
                            dist_param[variable]["sigma_bijector"],
                            # tfb.Identity()
                        ]
                    )

                if (dist_param[variable]["mu_d"] == "Gamma") or (
                    dist_param[variable]["sigma_d"] == "Gamma"
                ):
                    bijectors.append(tfb.Exp())
                else:
                    bijectors.append(tfb.Identity())

            else:
                variable = row["IDV"]
                if dist_param[variable]["fixed_d"] not in dist:
                    bijectors.append(dist_param[variable]["fixed_bijector"])
                else:
                    bijectors.append(tfb.Exp())

        if "sigma_target" not in dist_param:
            bijectors.append(tfb.Exp())

        return bijectors

    except Exception as e:
        LOG.warning("Error while getting bijectors")
        LOG.exception(e)
