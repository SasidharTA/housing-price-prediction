import re

import numpy as np
import pandas as pd
from formulae import design_matrices
from sklearn.base import BaseEstimator

from .BayesFramework.plot_utils import Plot
from .BayesFramework.Regression import BayesianEstimation


class TFPBayesian(BaseEstimator):
    def __init__(self, formula, model_params={}):
        if formula.split()[-1].strip() == ".":
            raise NotImplementedError(
                "TFP Bayesian not implemented with  above formula type"
            )

        self.model_params = {
            "objective": "regression",
            "sampler": "nuts",
            "num_chains": 4,
            "num_results": 2000,
            "num_burnin_steps": 1000,
            "num_leapfrog_steps": 10,
            "hmc_step_size": 0.01,
            "num_steps_between_results": 2,
            "max_energy_diff": 1000,
            "max_tree_depth": 10,
            "unrolled_leapfrog_steps": 1,
            "parallel_iterations": 10,
        }

        for params, value in model_params.items():
            self.model_params[params] = value

        self.framework_config_df = pd.DataFrame(
            list(self.model_params.items()), columns=["TagName", "Value"]
        )
        self.formula = formula

    def process_col(self, x):
        x = x.split("|")[0]
        x = re.sub(r"[^a-zA-Z0-9]", "_", x)
        return x

    def process_df(self, df):
        df1 = df.loc[:, ~df.T.duplicated(keep="last")]
        for col in df1.columns:
            if len(df1[col].unique()) == 1:
                df1.drop(col, inplace=True, axis=1)

        cols = df1.columns
        new_cols = [self.process_col(x) for x in cols]
        df1.columns = new_cols
        return df1

    def get_fixed_effects(self):
        self.fixed_effects = []
        self.fixed_feature_dict = {}

        for col in self.common_cols:
            if col == "Intercept":
                new_col = "global_intercept"
                self.fixed_feature_dict[new_col] = col
                self.fixed_effects.append("global_intercept")
            else:
                new_col = self.process_col(col)
                self.fixed_feature_dict[new_col] = col
                self.fixed_effects.append(new_col)

        return self

    def get_random_effects(self):
        self.random_effects = []
        self.random_feature_dict = {}
        for col in self.re_cols:
            if "1|" in col:
                new_col = col.replace("1|", "intercept_")
                self.random_feature_dict[new_col] = col
                self.random_effects.append(new_col)
            else:
                new_col = self.process_col(col)
                self.random_feature_dict[new_col] = col
                self.random_effects.append(new_col)

        return self

    def get_model_config_df(self):
        config_cols = [
            "DV",
            "IDV",
            "Include_IDV",
            "RandomEffect",
            "RandomFactor",
            "mu_d",
            "mu_d_loc_alpha",
            "mu_d_scale_beta",
            "sigma_d",
            "sigma_d_loc_alpha",
            "sigma_d_scale_beta",
            "mu_bijector",
            "sigma_bijector",
            "fixed_d",
            "fixed_d_loc_alpha",
            "fixed_d_scale_beta",
            "fixed_bijector",
        ]

        fixed_config_df = pd.DataFrame(
            index=range(0, len(self.fixed_effects)), columns=config_cols
        )
        if len(self.fixed_effects) != 0:
            fixed_config_df["DV"] = self.response_var
            fixed_config_df["IDV"] = self.fixed_effects
            fixed_config_df["Include_IDV"] = 1
            fixed_config_df["RandomEffect"] = 0

        for index in range(len(fixed_config_df)):
            feat = fixed_config_df.loc[index, "IDV"]
            key = self.fixed_feature_dict[feat]
            if key in list(self.priors_dict.keys()):
                prior_1 = self.priors_dict[key]
                keys_1 = list(prior_1)
                fixed_config_df.loc[index, "fixed_d"] = prior_1["dist"]
                fixed_config_df.loc[index, "fixed_d_loc_alpha"] = float(
                    prior_1[keys_1[1]]
                )
                fixed_config_df.loc[index, "fixed_d_scale_beta"] = float(
                    prior_1[keys_1[2]]
                )
            else:
                fixed_config_df.loc[index, "fixed_d"] = "Normal"
                fixed_config_df.loc[index, "fixed_d_loc_alpha"] = float(0)
                fixed_config_df.loc[index, "fixed_d_scale_beta"] = float(5)

            if key in list(self.bijectors_dict.keys()):
                fixed_config_df.loc[index, "fixed_bijector"] = self.bijectors_dict[key]
            else:
                fixed_config_df.loc[index, "fixed_bijector"] = "Identity"

        random_config_df = pd.DataFrame(
            index=range(0, len(self.random_effects)), columns=config_cols
        )

        if len(self.random_effects) != 0:
            random_config_df["DV"] = self.response_var
            random_config_df["IDV"] = self.random_effects
            random_config_df["Include_IDV"] = 1
            random_config_df["RandomEffect"] = 1
            random_config_df["RandomFactor"] = self.group_factor

        for index in range(len(random_config_df)):
            feat = random_config_df.loc[index, "IDV"]
            key = self.random_feature_dict[feat]
            if key in list(self.priors_dict.keys()):
                prior_1 = self.priors_dict[key]
                random_config_df.loc[index, "mu_d"] = prior_1["mu"]["dist"]
                random_config_df.loc[index, "mu_d_loc_alpha"] = float(
                    prior_1["mu"]["mu"]
                )
                random_config_df.loc[index, "mu_d_scale_beta"] = float(
                    prior_1["mu"]["sd"]
                )
                random_config_df.loc[index, "sigma_d"] = prior_1["sd"]["dist"]
                random_config_df.loc[index, "sigma_d_loc_alpha"] = float(
                    prior_1["sd"]["mu"]
                )
                random_config_df.loc[index, "sigma_d_scale_beta"] = float(
                    prior_1["sd"]["sd"]
                )
            else:
                random_config_df.loc[index, "mu_d"] = "Normal"
                random_config_df.loc[index, "mu_d_loc_alpha"] = float(0)
                random_config_df.loc[index, "mu_d_scale_beta"] = float(5)
                random_config_df.loc[index, "sigma_d"] = "HalfCauchy"
                random_config_df.loc[index, "sigma_d_loc_alpha"] = float(0)
                random_config_df.loc[index, "sigma_d_scale_beta"] = float(2)

            if key in list(self.bijectors_dict.keys()):
                random_config_df.loc[index, "mu_bijector"] = self.bijectors_dict[key][
                    "mu"
                ]
                random_config_df.loc[index, "sigma_bijector"] = self.bijectors_dict[
                    key
                ]["sd"]
            else:
                random_config_df.loc[index, "mu_bijector"] = "Identity"
                random_config_df.loc[index, "sigma_bijector"] = "Exp"

        self.model_config_df = pd.concat(
            [random_config_df, fixed_config_df]
        ).reset_index(drop=True)
        cols_convert = [
            "mu_d_loc_alpha",
            "mu_d_scale_beta",
            "sigma_d_loc_alpha",
            "sigma_d_scale_beta",
            "fixed_d_loc_alpha",
            "fixed_d_scale_beta",
        ]

        self.model_config_df[cols_convert] = self.model_config_df[cols_convert].apply(
            pd.to_numeric, errors="coerce", axis=1
        )

        return self.model_config_df

    def fit(self, data, priors={}, bijectors={}):
        self.priors_dict = priors
        self.bijectors_dict = bijectors
        self.dm = design_matrices(self.formula, data)

        if self.dm.common != None:
            X = self.dm.common.as_dataframe()
            self.common_cols = list(X.columns)
        else:
            self.common_cols = []
            X = pd.DataFrame()

        if self.dm.group != None:
            group_factor = []
            group_expr_data = []
            term_z_index = {}
            i = 0
            for key, gterm in self.dm.group.terms.items():
                # print(gterm)
                group_factor.append(gterm.factor.name)
                group_expr_data.append(
                    self.dm.group.design_matrix[:, self.dm.group.slices[key]].sum(
                        axis=1
                    )
                )
                term_z_index[key] = i
                i += 1
            group_factor = list(set(group_factor))

            if len(group_factor) > 1:
                raise NotImplementedError(
                    "TFP Bayesian is not implemented for multiple group factors"
                )
            else:
                group_factor = group_factor[0]

            self.re_cols = list(term_z_index.keys())
            Z = pd.DataFrame(np.array(group_expr_data).T, columns=self.re_cols)
            clusters = self.dm.data[group_factor].astype(str)
            self.group_factor = group_factor
        else:
            self.re_cols = []
            Z = pd.DataFrame()

        self.response_var = self.dm.response.name
        y = np.array(self.dm.response)

        df = pd.concat([X, Z], axis=1)
        df[self.response_var] = y

        self.model_data = self.process_df(df)

        if self.dm.group != None:
            self.model_data[self.group_factor] = clusters
            self.model_data[self.group_factor] = self.model_data[
                self.group_factor
            ].astype(object)

        self.get_fixed_effects()
        self.get_random_effects()
        self.get_model_config_df()

        self.tfp_model = BayesianEstimation(
            self.model_data, self.model_config_df, self.framework_config_df
        )
        self.tfp_model.train()
        self.is_fitted_ = True
        return self.tfp_model.dict_sample

    def get_summary(self):
        return self.tfp_model.summary()

    def predict(self, data_new):

        if self.dm.common != None:
            common_eval = self.dm.common.evaluate_new_data(data_new)
            X = common_eval.as_dataframe()
        else:
            X = pd.DataFrame()

        if self.dm.group != None:
            group_eval = self.dm.group.evaluate_new_data(data_new)
            group_expr_data = []
            term_z_index = {}
            i = 0
            for key, gterm in group_eval.terms.items():
                # print(gterm)
                group_expr_data.append(
                    group_eval.design_matrix[:, group_eval.slices[key]].sum(axis=1)
                )
                term_z_index[key] = i
                i += 1
            Z = pd.DataFrame(np.array(group_expr_data).T, columns=self.re_cols)
            clusters = data_new[self.group_factor].astype(str)
        else:
            Z = pd.DataFrame()

        y = np.array(data_new[self.response_var])

        df = pd.concat([X, Z], axis=1)

        df[self.response_var] = y
        data_pr = self.process_df(df)

        if self.dm.group != None:
            data_pr[self.group_factor] = clusters
            data_pr[self.group_factor] = data_pr[self.group_factor].astype(object)

        y_pred, metrics = self.tfp_model.predict(data_pr=data_pr)
        return y_pred

    def save_plots(self, output_folder_path=None):
        plf = Plot(self.tfp_model)
        plf.save_all_plots()
