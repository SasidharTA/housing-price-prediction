"""Regression module lets you apply Bayesian Regression model for a given set of random and fixed effect variables."""

import collections
import copy
import os
import pickle  # noqa: S403
import sys
import time
import warnings

import arviz as az
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from .advi import get_bijector, run_approximation
from .joint_distribution_coroutine import create_joint_dist_co
from .logging_utils import get_logger
from .predict import prediction
from .utils import (
    check_samples,
    check_values,
    get_tensor_cat,
    get_tensor_var,
    group_estimates,
    tensor_to_dictionary,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

LOG = get_logger(__name__)

warnings.filterwarnings("ignore")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

thismodule = sys.modules[__name__]

tfd = tfp.distributions
tfb = tfp.bijectors


class BayesianEstimation:
    """
    Partially pooled heirarchical bayesian model for a given set of random and fixed effects.

    Parameters
    ----------
    data_df : pd.DataFrame
    model_config_df : pd.DataFrame
        Pandas DataFrame made from config
            * `Include_IDV :` 1/0
            * `RandomEffect :` Random Effect=1, Fixed Effect=0
            * `RandomFactor :` Grouping Column
            * `mu_d :` distribution
            * `mu_d_loc_alpha :` <int/float>
            * `mu_d_scale_beta :` <int/float>
            * `sigma_d :` distribution
            * `sigma_d_loc_alpha :` <int/float>
            * `sigma_d_scale_beta :` <int/float>
            * `mu_bijector, sigma_bijector :` Identity/Exp
            * `fixed_d fixed_d_loc_alpha :` <int/float>
            * `fixed_d_scale_beta :` <int/float>
            * `fixed_bijector :` Identity/Exp
    framework_config_df:pd.DataFrame
        Pandas DataFrame made from config
            * `objective :` regression
            * `sampler :` hmc/nuts/VI
            * `num_chains :` <int> number of mcmc chains (passed to hmc.sample_chains())
            * `num_results :` <int> number of results
            * `num_burnin_steps :` <float> parameter to TFP Hmc sampler
            * `num_leapfrog_steps :` <float> parameter to TFP Hmc sampler
            * `hmc_step_size :` <float> parameter to TFP Hmc sampler
            * `num_steps_between_results :` <int> Thinning rate
            * `max_energy_diff :` <float> parameter to nuts sampler
            * `max_tree_depth :` <int> parameter to nuts sampler
            * `unrolled_leapfrog_steps :` <int> parameter to nuts sampler
            * `parallel_iterations :` <int> parameter to nuts sampler
            * `num_variational_steps :` <float> parameter to VI

    Returns
    -------
        None

    Notes
    -----
        * estimates of all random effect params with Rhat values group wise.
        * estimates of all fixed effect paramas with Rhat values on population
        * Traceplots: as per the arguments
        * Full sample trace: as per the arguments
        * Functions for traceplot and posterior plot.
    """

    def __init__(  # noqa: C901
        self,
        data_df_original,
        model_config_df_original,
        framework_config_df,
        pickle_file="",
    ):

        data_df = data_df_original.copy()
        model_config_df = model_config_df_original.copy()

        if "global_intercept" not in data_df.columns:
            data_df["global_intercept"] = 1

        self.join_dist_list = []
        t = time.localtime()
        self.dt = time.strftime("%b-%d-%Y_%H%M", t)

        model_config_df = model_config_df[model_config_df["Include_IDV"] == 1].copy()
        model_config_df.drop("Include_IDV", axis=1, inplace=True)
        # LOG.info("Removing special characters from dv and idvs.")
        # model_config_df.loc[:, "DV"] = rem_specialchar_array(model_config_df["DV"])
        # model_config_df.loc[:, "IDV"] = rem_specialchar_array(model_config_df["IDV"])
        model_config_df.loc[:, "RandomFactor"][
            model_config_df["RandomFactor"].isna()
        ] = ""

        # model_config_df.loc[:, "RandomFactor"] = rem_specialchar_array(
        #     model_config_df["RandomFactor"]
        # )
        self.model_config_df = model_config_df

        # data_df.columns = rem_specialchar_array(data_df.columns)

        mapped_list = []
        for column in [
            s for s in model_config_df["RandomFactor"].unique() if s not in [np.nan, ""]
        ]:
            data_df = data_df[data_df[column].notna()]
            column_name = sorted(data_df[column].unique())
            data_df[column + "_original"] = data_df[column]
            data_df[column] = (
                data_df[column]
                .astype(pd.api.types.CategoricalDtype(categories=column_name))
                .cat.codes
            )
            data_df[column] = data_df[column].astype(int)
            mapped_list.append(column)
            mapped_list.append(column + "_original")
        self.data_df = data_df

        self.ModelParamsTuple = None
        self.SamplesTuple = None
        self.duplicate_data = None
        self.dict_sample = None
        self.approx_param = None
        self.option = None
        self.loss_ = None

        hyperparam_dict = {}
        for _, row in framework_config_df.iterrows():
            if row["TagName"] not in ["objective", "add_globalintercept"]:
                hyperparam_dict[row["TagName"]] = row["Value"]

        default_tags = {
            "num_chains": 4,
            "num_results": 2000,
            "num_burnin_steps": 1000,
            "num_leapfrog_steps": 10,
            "hmc_step_size": 0.01,
            "num_steps_between_results": 2,
            "max_energy_diff": 1000.0,
            "max_tree_depth": 10,
            "unrolled_leapfrog_steps": 1,
            "parallel_iterations": 10,
            "num_variational_steps": 300,
        }

        for k, v in default_tags.items():
            if k not in hyperparam_dict:
                hyperparam_dict[k] = v
            else:
                if np.isnan(hyperparam_dict[k]):
                    hyperparam_dict[k] = v

        spec_dict = {
            "dv": model_config_df["DV"].iloc[0],
            "idvs": [
                item.replace("intercept_", "")
                for item in list(model_config_df["IDV"].drop_duplicates())
                if item not in ["sigma_target"]
            ],
            "group_cols": [
                s
                for s in model_config_df["RandomFactor"].unique()
                if s not in [np.nan, ""]
            ],
            "hyperparams": hyperparam_dict,
        }
        self.spec_dict = spec_dict
        LOG.info("Printing Hyperparams:")
        LOG.info(self.spec_dict["hyperparams"])

        data_df = self.data_df
        self.mapped_df = data_df[mapped_list]
        data_df = data_df[
            list(
                set(
                    [self.spec_dict["dv"]]
                    + self.spec_dict["idvs"]
                    + self.spec_dict["group_cols"]
                )
            )
        ]
        data_df = data_df.dropna()
        self.data_df = data_df

        dist_param = {}
        num_chains = spec_dict["hyperparams"]["num_chains"]  # noqa: F841
        for _, row in model_config_df.iterrows():
            if row["IDV"] == "global_intercept":
                var = "global_intercept"
                tmp_param = {
                    "fixed_d": row["fixed_d"],
                    "fixed_d_loc": row["fixed_d_loc_alpha"],
                    "fixed_d_scale": row["fixed_d_scale_beta"],
                    "fixed_bijector": row["fixed_bijector"],
                }
                dist_param[var] = tmp_param

            elif "intercept" in row["IDV"]:
                var = row["RandomFactor"]
                tmp_param = {
                    "mu_d": row["mu_d"],
                    "mu_d_loc": row["mu_d_loc_alpha"],
                    "mu_d_scale": row["mu_d_scale_beta"],
                    "sigma_d": row["sigma_d"],
                    "sigma_d_loc": row["sigma_d_loc_alpha"],
                    "sigma_d_scale": row["sigma_d_scale_beta"],
                    "mu_bijector": row["mu_bijector"],
                    "sigma_bijector": row["sigma_bijector"],
                }
                dist_param[var] = tmp_param

            else:
                var = row["IDV"]
                if row["RandomEffect"] == 1:
                    tmp_param = {
                        "mu_d": row["mu_d"],
                        "mu_d_loc": row["mu_d_loc_alpha"],
                        "mu_d_scale": row["mu_d_scale_beta"],
                        "sigma_d": row["sigma_d"],
                        "sigma_d_loc": row["sigma_d_loc_alpha"],
                        "sigma_d_scale": row["sigma_d_scale_beta"],
                        "mu_bijector": row["mu_bijector"],
                        "sigma_bijector": row["sigma_bijector"],
                    }
                else:
                    tmp_param = {
                        "fixed_d": row["fixed_d"],
                        "fixed_d_loc": row["fixed_d_loc_alpha"],
                        "fixed_d_scale": row["fixed_d_scale_beta"],
                        "fixed_bijector": row["fixed_bijector"],
                    }
                dist_param[var] = tmp_param
        dist_bij = copy.deepcopy(dist_param)
        self.dist_bij = dist_bij
        LOG.info("Printing distribution and bijectors:")
        LOG.info(self.dist_bij)

        for key1, value1 in dist_param.items():
            for key2, value2 in value1.items():
                if "bijector" in key2:
                    if value2 == "Identity":
                        dist_param[key1][key2] = tfb.Identity()
                    elif value2 == "Exp":
                        dist_param[key1][key2] = tfb.Exp()
        self.dist_param = dist_param

        random_effect = []
        fixed_effect = []

        for _, row in self.model_config_df.iterrows():
            if row["RandomEffect"] == 0:
                if (row["IDV"] != "global_intercept") and (
                    row["IDV"] != "sigma_target"
                ):
                    fixed_effect.append(row["IDV"])
            elif row["RandomEffect"] == 1:
                check = "intercept_" + row["RandomFactor"]
                l2 = []
                if row["IDV"] != check:
                    l2.append(row["IDV"])
                    l2.append(row["RandomFactor"])
                    random_effect.append(l2)
        self.random_effect = random_effect
        self.fixed_effect = fixed_effect

        self.run_id = None

        data = {}
        data["idvs"] = self.spec_dict["idvs"]
        data["dv"] = self.spec_dict["dv"]
        data["random_effects"] = self.random_effect
        data["fixed_effects"] = self.fixed_effect
        data["group_column"] = self.spec_dict["group_cols"]

        if pickle_file:
            LOG.info("Using given pickle file.")
            pickle_off = open(pickle_file, "rb")
            all_data = {}
            all_data = pickle.load(pickle_off)  # noqa: S301
            samples = all_data["samples"]
            acceptance_probs = all_data["acceptance_probs"]

            self.join_dist_list = all_data["join_dist_list"]
            self.ModelParamsTuple = collections.namedtuple(
                "ModelParams", self.join_dist_list[:-1]
            )
            LOG.info("got model param tuple")
            self.SamplesTuple = self.ModelParamsTuple._make(samples)
            LOG.info("got samples tuple")

            # print("Acceptance Probabilities: ", acceptance_probs.numpy())
            LOG.info("Acceptance Probabilities: ")
            LOG.info(acceptance_probs.numpy())
            try:

                for var in self.join_dist_list[:-1]:
                    if "mu_" in var or "sigma_" in var:
                        LOG.info(
                            "R-hat for ",
                            var,
                            "\t: ",
                            tfp.mcmc.potential_scale_reduction(
                                getattr(self.SamplesTuple, var)
                            ).numpy(),
                        )

            except Exception as e:
                LOG.error("------Error while calculating r-hat-----")
                LOG.exception(e)

    def preprocess(self):
        """
        To Convert spec_dict variables to tensor dv,idv and Categorical data.

        Raises
        ------
        TypeError:
            An error occurred while converting dv/idv into tensor.

        Returns
        -------
        tensor_d : dictionary
            columns Converted to Tensors
        """
        LOG.info("Converting dv and idvs into tensor.")
        tensor_d = {}
        for key, value in self.spec_dict.items():
            if key == "dv":
                try:
                    tensor_d[value] = get_tensor_var(value, self.data_df)
                except Exception as e:
                    LOG.error("An error occurred while converting dv into tensor.")
                    LOG.exception(e)

            elif key == "idvs":
                for idv in value:
                    idv_1 = idv
                    try:
                        if idv_1 not in self.spec_dict["group_cols"]:
                            tensor_d[idv_1] = get_tensor_var(idv_1, self.data_df)

                        else:
                            tensor_d[idv_1] = get_tensor_cat(idv_1, self.data_df)

                    except Exception as e:
                        LOG.error(
                            "An error occurred while converting idvs into tensor."
                        )
                        LOG.exception(e)

        return tensor_d

    def select_sampling_technique(
        self,
        condition,
        initial_state,
        unconstraining_bijectors,
        joint_dist_model_log_prob,
        seed,
    ):
        """
        To select the sampling technique and assign the initial state.

        Parameters
        ----------
        condition: str
            hmc or nuts
        initial_state: tensor
            Tensors representing the initial state
        unconstraining_bijectors: str
            Identity or Exp
        joint_dist_model_log_prob: probability density
            Log probability density
        seed: int/float
            random generator

        Returns
        -------
        kernel: tf object
            kernel for the selected Method
        """
        try:
            if condition == "hmc":
                hmc_step_size = self.spec_dict["hyperparams"]["hmc_step_size"]
                num_leapfrog_steps = self.spec_dict["hyperparams"]["num_leapfrog_steps"]

                sampler = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=joint_dist_model_log_prob,
                    num_leapfrog_steps=num_leapfrog_steps,
                    step_size=hmc_step_size,
                )
                kernel = tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=sampler, bijector=unconstraining_bijectors
                )
                LOG.info("hmc sampler is selected.")

            else:
                num_chains = self.spec_dict["hyperparams"]["num_chains"]
                hmc_step_size = self.spec_dict["hyperparams"]["hmc_step_size"]
                num_burnin_steps = self.spec_dict["hyperparams"]["num_burnin_steps"]
                max_energy_diff = self.spec_dict["hyperparams"]["max_energy_diff"]
                max_tree_depth = self.spec_dict["hyperparams"]["max_tree_depth"]
                unrolled_leapfrog_steps = self.spec_dict["hyperparams"][
                    "unrolled_leapfrog_steps"
                ]
                parallel_iterations = self.spec_dict["hyperparams"][
                    "parallel_iterations"
                ]

                target_accept_prob = 0.8
                num_adaptation_steps = int(0.8 * num_burnin_steps)

                step_size = [
                    tf.fill(
                        [num_chains] + [1] * (len(s.shape) - 1),
                        tf.constant(hmc_step_size, np.float32),
                    )
                    for s in initial_state
                ]
                sampler = tfp.mcmc.NoUTurnSampler(
                    joint_dist_model_log_prob,
                    step_size=step_size,
                    # seed=seed,   # seed argument has been depricated and removed. Moved to sample_chain() method
                    max_energy_diff=max_energy_diff,
                    max_tree_depth=max_tree_depth,
                    unrolled_leapfrog_steps=unrolled_leapfrog_steps,
                    parallel_iterations=parallel_iterations,
                )
                kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                    tfp.mcmc.TransformedTransitionKernel(
                        inner_kernel=sampler, bijector=unconstraining_bijectors
                    ),
                    target_accept_prob=target_accept_prob,
                    num_adaptation_steps=num_adaptation_steps,
                    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                        inner_results=pkr.inner_results._replace(
                            step_size=new_step_size
                        )
                    ),
                    step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                    log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
                )
                LOG.info("nuts sampler is selected.")
        except Exception as e:
            LOG.error("---Error occured while getting sampler/kernel.-----")
            LOG.exception(e)
            sys.exit(1)

        return kernel

    def get_ready(
        self, unconstraining_bijectors, joint, joint_dist_model_log_prob, seed
    ):
        """
        To start the sampling function with all the initial parameter.

        Parameters
        ----------
            seed: int
                random generator
            unconstraining_bijectors: list
                bijectors list
            joint_dist_model_log_prob: numeric
                log probability density

        Returns
        -------
            sampling_technique
            kernel
            initial_state: tensor
            unconstraining_bijectors: str
            seed: int
            num_chains: tf.int32
            num_results: tf.int32
            num_burnin_steps: tf.int32
            num_steps_between_results: tf.int32
        """
        num_chains = tf.cast(self.spec_dict["hyperparams"]["num_chains"], tf.int32)
        num_results = int(self.spec_dict["hyperparams"]["num_results"])
        num_burnin_steps = int(self.spec_dict["hyperparams"]["num_burnin_steps"])
        num_steps_between_results = int(
            self.spec_dict["hyperparams"]["num_steps_between_results"]
        )

        initial_state_ = joint.sample(sample_shape=20, seed=seed)[:-1]
        LOG.info("Printing initial_state: ")
        LOG.info(initial_state_)

        initial_state = []
        for i in range(len(initial_state_)):
            initial_state.append(
                tf.repeat(
                    tf.expand_dims(tf.reduce_mean(initial_state_[i], 0), axis=0),
                    repeats=num_chains,
                    axis=0,
                )
            )

        check_values(initial_state)
        sampling_technique = self.spec_dict["hyperparams"]["sampler"]
        kernel = self.select_sampling_technique(
            sampling_technique,
            initial_state,
            unconstraining_bijectors,
            joint_dist_model_log_prob,
            seed=seed,
        )

        return (
            sampling_technique,
            kernel,
            initial_state,
            unconstraining_bijectors,
            num_chains,
            num_results,
            num_burnin_steps,
            num_steps_between_results,
        )

    @tf.function(jit_compile=True)
    @tf.autograph.experimental.do_not_convert
    def sample_model(
        self,
        sampling_technique,
        kernel,
        initial_state,
        unconstraining_bijectors,
        num_chains,
        num_results,
        num_burnin_steps,
        num_steps_between_results,
        seed,
    ):
        """
        To get the Samples from the model.

        Parameters
        ----------
            condition: str
                hmc or nuts.
            initial_state: tensor
                Tensors representing the initial state.
            unconstraining_bijectors: str
            seed: int
                random generator.
            num_chains: tf.int32
            num_results: tf.int32
            num_burnin_steps: tf.int32

        Returns
        -------
            samples: list(tensor)
            acceptance_probs: list(tensor)
        """
        LOG.info("Started Sampling ")
        try:
            samples, kernel_results = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_state,
                kernel=kernel,
                num_steps_between_results=num_steps_between_results,
                seed=seed,
            )
        except Exception as e:
            LOG.error(
                "Error while generating samples. Please try with different priors."
            )
            LOG.exception(e)
            sys.exit(1)

        LOG.info("Calculating acceptance probs")
        try:
            if sampling_technique == "hmc":
                acceptance_probs = tf.reduce_mean(
                    tf.cast(kernel_results.inner_results.is_accepted, tf.float32),
                    axis=0,
                )

            else:
                acceptance_probs = tf.reduce_mean(
                    tf.cast(
                        kernel_results.inner_results.inner_results.is_accepted,
                        tf.float32,
                    ),
                    axis=0,
                )

        except Exception as e:
            LOG.error("----Error while getting acceptance prob.----")
            LOG.exception(e)

        return samples, acceptance_probs

    def train(self, fixed_seed=123):  # noqa: C901
        """
        To create and Execute the Joint Distribution Sequence and Get the Model Metrics Results after Sampling.

        Parameters
        ----------
            fixed_seed: int
                random generator

        Returns
        -------
            Prints Model Metrics
        """
        start_time = time.time()
        tensor_d = self.preprocess()
        LOG.info("preprocess done")
        unconstraining_bijectors = []

        try:
            LOG.info("creating joint distribution coroutine")
            joint = create_joint_dist_co(
                self.model_config_df,
                self.data_df,
                self.join_dist_list,
                self.dist_param,
                unconstraining_bijectors,
                tensor_d,
            )

        except Exception as e:
            LOG.warning("---error occured while executing joint_dist_seq----")
            LOG.exception(e)

        def joint_dist_model_log_prob(*args):
            return joint.log_prob(args + (tensor_d[self.join_dist_list[-1]],))

        if self.spec_dict["hyperparams"]["sampler"] == "VI":
            LOG.info("Using variational inference")
            self.option = self.spec_dict["hyperparams"]["sampler"]

            bijectors = get_bijector(self.model_config_df, self.dist_param)

            sample_event_tensor = joint.event_shape_tensor()[:-1]
            surrogate_posterior = (
                tfp.experimental.vi.build_factored_surrogate_posterior(
                    event_shape=sample_event_tensor, bijector=bijectors
                )
            )

            opt = tf.optimizers.Adam(learning_rate=0.1)
            num_variational_steps = int(
                self.spec_dict["hyperparams"]["num_variational_steps"]
            )

            loss_ = run_approximation(
                joint_dist_model_log_prob,
                surrogate_posterior,
                opt,
                num_variational_steps,
                fixed_seed,
            )
            self.loss_ = loss_

            # getting mu and sd value
            approx_param = dict()
            posterior_samples = surrogate_posterior.sample(50)
            for i, rvname in enumerate(self.join_dist_list[:-1]):
                approx_param[rvname] = {
                    "mu": np.mean(posterior_samples[i], axis=0),
                    "sd": np.std(posterior_samples[i], axis=0),
                }
            # free_param = surrogate_posterior.trainable_variables
            # for i, rvname in enumerate(self.join_dist_list[:-1]):
            #     approx_param[rvname] = {
            #         "mu": free_param[i * 2].numpy(),
            #         "sd": free_param[i * 2 + 1].numpy(),
            #     }

            self.approx_param = approx_param

        else:

            (
                sampling_technique,
                kernel,
                initial_state,
                unconstraining_bijectors,
                num_chains,
                num_results,
                num_burnin_steps,
                num_steps_between_results,
            ) = self.get_ready(
                unconstraining_bijectors,
                joint,
                joint_dist_model_log_prob,
                seed=fixed_seed,
            )

            samples, acceptance_probs = self.sample_model(
                sampling_technique,
                kernel,
                initial_state,
                unconstraining_bijectors,
                num_chains,
                num_results,
                num_burnin_steps,
                num_steps_between_results,
                seed=fixed_seed,
            )

            self.ModelParamsTuple = collections.namedtuple(
                "ModelParams", self.join_dist_list[:-1]
            )

            self.SamplesTuple = self.ModelParamsTuple._make(samples)
            LOG.info("Creating samples tuple")
            all_data = {}
            all_data["samples"] = samples
            all_data["acceptance_probs"] = acceptance_probs
            all_data["join_dist_list"] = self.join_dist_list

            dict_sample = tensor_to_dictionary(
                self.join_dist_list, self.SamplesTuple, self.mapped_df, self.spec_dict
            )
            self.dict_sample = dict_sample

            check_samples(self.dict_sample)

            LOG.info("Acceptance Probabilities: ")
            LOG.info(acceptance_probs.numpy())
            try:

                for var in self.join_dist_list[:-1]:
                    if "mu_" in var or "sigma_" in var:
                        rhat_value = tfp.mcmc.potential_scale_reduction(
                            getattr(self.SamplesTuple, var)
                        ).numpy()
                        LOG.info(f"R-hat for {var} : {rhat_value}")

            except Exception as e:
                LOG.error("------Error while calculating r-hat-----")
                LOG.exception(e)

        run_time = time.time() - start_time
        pass

    def summary(self):
        """
        To Save Summary for indivudual groups.

        Parameters
        ----------
        Returns
        -------
            Returns group summary in the directory mentioned.

        Raises
        ------
            error if object not present for saving.
        """

        try:
            if self.option == "VI":
                dict_estimates = group_estimates(
                    self.approx_param, self.mapped_df, self.spec_dict
                )
                summary = pd.DataFrame.from_dict(dict_estimates, orient="index")

            else:
                summary = az.summary(self.dict_sample)

        except Exception as e:
            LOG.error("----Error while saving Group Summary.----")
            LOG.exception(e)

        return summary

    def predict(self, data_pr):
        """
        To Predict the values of the target variable and save the predicted values with original dataset.

        Parameters
        ----------
        data_pr : dataframe
            dataset for prediction.

        Returns
        -------
        y_pred : float
            predicted values.
        r2_value : float
            r2_score.
        RMSE : float
        mape : float
        mae : float
        wmape : float
        """

        y_pred, metrics = prediction(
            data_pr,
            self.join_dist_list,
            self.SamplesTuple,
            self.spec_dict,
            self.mapped_df,
            self.fixed_effect,
            self.dt,
            self.random_effect,
            self.option,
            self.approx_param,
        )

        return y_pred, metrics
