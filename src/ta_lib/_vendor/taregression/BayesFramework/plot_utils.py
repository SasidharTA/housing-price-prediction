"""Plot_utils module lets you plot trace and posterior plots."""

import os
import sys
import warnings

import arviz as az
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages

# import logging
from .logging_utils import get_logger

warnings.filterwarnings("ignore")

thismodule = sys.modules[__name__]
LOG = get_logger(__name__)


class Plot:
    """Functions for traceplot and posterior plot.

    Parameters
    ----------
    obj : BayesianEstimation class object

    """

    def __init__(self, obj, output_folder_path=None):

        self.dt = obj.dt
        self.join_dist_list = obj.join_dist_list
        self.SamplesTuple = obj.SamplesTuple
        self.dict_sample = obj.dict_sample
        self.option = obj.option
        self.loss_ = obj.loss_
        if output_folder_path == None:
            raise ValueError("Output folder path is None, must pass a str")
        else:
            self.output_path = output_folder_path

    def plot_posterior(self, var_name, var_samples):
        """
        Plot posterior for variable.

        Parameters
        ----------
        var_name : string
            Name of the variable to plot.
        var_samples : tensor
            Contains sample.

        Returns
        -------
        fig : posterior plot of variable.

        """
        try:
            if isinstance(var_samples, tf.Tensor):
                var_samples = var_samples.numpy()

            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
            ax.hist(var_samples.flatten(), bins=40, edgecolor="white")
            sample_mean = var_samples.mean()
            ax.text(
                sample_mean,
                100,
                "mean={:.3f}".format(sample_mean),
                color="white",
                fontsize=12,
            )
            ax.set_xlabel("posterior of " + var_name)
            return fig

        except Exception as e:
            LOG.warning(
                "----Error while plotting posteriors for " + var_name + "----------"
            )
            LOG.exception(e)

    def plot_posterior_allvars(
        self,
        output_folder_path=None,
    ):
        """
        Execute `plot_posterior` function for every variable.

        Parameters
        ----------
        output_folder_path : string, optional
            Path of output folder. The default is 'output/bayesian_model_train_summary/'.

        Returns
        -------
        None.

        """
        if output_folder_path is None:
            output_folder_path = os.path.join(
                self.output_path, "bayesian_model_train_summary"
            )
        if not os.path.exists(output_folder_path):
            LOG.info("Creating " + output_folder_path + " path")
            os.makedirs(output_folder_path)

        var_list = [
            s
            for s in self.join_dist_list[:-1]
            if (("mu_" not in s) & ("sigma_" not in s))
        ]
        file_post = os.path.join(
            output_folder_path, "plot_posterior_" + self.dt + ".pdf"
        )
        post = PdfPages(file_post)
        LOG.info("Generating posterior plots.")
        for var in var_list:
            plot_post = self.plot_posterior(var, getattr(self.SamplesTuple, var))
            post.savefig(plot_post)
        post.close()

    def plot_trace(
        self,
        var=[],
        plot_v="trace",
        output_folder_path=None,
    ):
        """
        Function for trace plot using arviz.

        Parameters
        ----------
        plot_v : string, optional
             The default is 'trace'.
        output_folder_path : string, optional
            Path of output folder. The default is 'output/bayesian_model_train_summary/'.

        Returns
        -------
        None.

        """
        if output_folder_path is None:
            output_folder_path = os.path.join(
                self.output_path, "bayesian_model_train_summary"
            )
        if not os.path.exists(output_folder_path):
            LOG.info("Creating " + output_folder_path + " path")
            os.makedirs(output_folder_path)

        try:
            az_trace = az.from_dict(posterior=self.dict_sample)

            file = os.path.join(
                output_folder_path, self.dt + "plot_" + plot_v + "_" + self.dt + ".pdf"
            )
            pp = PdfPages(file)

            if var:
                for name in var:
                    if name in self.dict_sample:
                        LOG.info("Generating trace plots for " + name)
                        az.plot_trace(az_trace, var_names=name, figsize=(10, 3))
                    else:
                        LOG.exception("Unable to find any samples/traces for " + name)

            else:
                LOG.info("Generating trace plots.")
                for key, value in self.dict_sample.items():
                    plot_s = (
                        "az.plot_"
                        + plot_v
                        + "(az_trace, var_names=key, figsize=(10,3))"
                    )
                    try:
                        plot = exec(plot_s)  # noqa: S102
                        pp.savefig(plot)
                    except Exception as e:  # noqa: F841
                        LOG.warning("Error while generating trace plots for " + key)
                pp.close()
        except Exception as e:
            LOG.warning("----Error while generating trace plots.----")
            LOG.exception(e)

    def loss_graph(
        self,
        output_folder_path=None,
    ):
        """
        Save loss graph in variational inference method.

        Parameters
        ----------
        output_folder_path : string, optional
            Path of output folder. The default is 'output/bayesian_model_train_summary/'.

        Returns
        -------
        None.

        """
        if output_folder_path is None:
            output_folder_path = os.path.join(
                self.output_path, "bayesian_model_train_summary"
            )
        if not os.path.exists(output_folder_path):
            LOG.info("Creating " + output_folder_path + " path")
            os.makedirs(output_folder_path)

        try:
            file_loss = os.path.join(
                output_folder_path, "loss_graph_" + self.dt + ".pdf"
            )
            loss_g = PdfPages(file_loss)
            f = plt.figure()
            plt.plot(self.loss_, "k-")
            plt.xlabel("iter")
            plt.ylabel("loss")
            plt.title("loss during training")
            loss_g.savefig(f)
            loss_g.close()

        except Exception as e:
            LOG.warning("----Error while generating loss graph.----")
            LOG.exception(e)

    def save_all_plots(
        self,
        output_folder_path=None,
    ):
        """
        Executes plot_trace() and plot_posterior_allvars().

        Parameters
        ----------
        output_folder_path : string, optional
            path where the outputs are to be saved. The default is "output/bayesian_model_train_summary/".

        Returns
        -------
        None.

        """
        if output_folder_path is None:
            output_folder_path = os.path.join(
                self.output_path, "bayesian_model_train_summary"
            )
        if not os.path.exists(output_folder_path):
            LOG.info("Creating " + output_folder_path + " path")
            os.makedirs(output_folder_path)

        if self.option == "VI":
            LOG.info(
                "Can't generate traceplot and posterior plots in variational inference method."
            )
            self.loss_graph(output_folder_path=output_folder_path)

        else:
            self.plot_trace(output_folder_path=output_folder_path)
            self.plot_posterior_allvars(output_folder_path=output_folder_path)
