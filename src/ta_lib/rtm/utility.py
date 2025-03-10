"""Module for listing down additional custom functions required for the notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_flavor as pf  # noqa
import re  # noqa
import seaborn as sns
import statsmodels.api as sm  # noqa
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ta_lib.core import dataset  # noqa
from ta_lib.regression.api import SKLStatsmodelOLS

# sales impact modelling related functions


def coefficient_check(formula, data, neg_impact_vars, response, correl_data):
    """Compute a coefficient table for a linear regression model and adjusts the signs of the coefficients based on some rules.

    The function fits a linear regression model specified by the formula parameter and the data parameter, using the statsmodels library. It then extracts the coefficients of the model and creates a table with them, in the order they appear in the formula. The first coefficient is dropped from the table, as it corresponds to the intercept term.
    The function then adjusts the sign of some coefficients based on the neg_impact_vars parameter. For each feature name in neg_impact_vars, the function identifies all coefficients in the table that contain that name and sets their sign to negative. All other coefficients have their sign set to positive.
    Finally, the function adds a column to the table called 'corr_sign' that indicates the sign of the correlation between each feature and the response variable, as given by the correl_data parameter.


    Parameters
    ----------
    formula : str
              A formula specifying the linear regression model to be fitted.
    data : pandas.DataFrame
           The data used to fit the linear regression model.
    neg_impact_vars : List[str]
                      A list of feature names that should have negative coefficients in the fitted model.
    response : str
               The name of the response variable in the data.
    correl_data : pandas.DataFrame
                  A correlation matrix of the features in the data.

    Returns
    -------
    coeff_table : pandas.DataFrame
                  A table with the coefficients of the linear regression model, with some of them possibly having their sign adjusted based on the neg_impact_vars parameter.

    """

    coeff_table = (
        smf.ols(formula, data).fit().params.reset_index().rename(columns={0: "coef"})
    )
    coeff_table = coeff_table.drop(0, axis=0)
    coeff_table["sign"] = 1
    coeff_table["corr_sign"] = coeff_table["index"].map(
        dict(zip(correl_data["index"][1:], correl_data[response][1:]))
    )

    for col in neg_impact_vars:
        neg_coef = coeff_table["index"][coeff_table["index"].str.contains(col)].tolist()
        pos_coef = coeff_table["index"][  # noqa
            ~coeff_table["index"].str.contains(col)
        ].tolist()[1:]
        # print(coeff_table)
        coeff_table["sign"] = np.where(coeff_table["index"].isin(neg_coef), -1, 1)

    return coeff_table


def stepwise_regression(
    data, response, neg_coeff_columns, necessary_col, correlation_data
):
    """Perform a stepwise regression on the data, selecting features based on their p-value and the sign of their coefficient.

    The function performs a stepwise regression on the data, starting with the features specified in necessary_col. At each step, it considers adding a new candidate feature to the model, and computes the p-value of the model fit with that candidate feature. If the p-value is significant (less than or equal to 0.1), and if the signs of the coefficient and correlation of the candidate feature are intuitive (i.e. negative for features in neg_coeff_columns, and the same sign as the correlation with the response variable), the candidate feature is added to the selected features.
    The function stops adding features to the model when all remaining candidates have a p-value greater than 0.1, or when no candidate features remain. The final model is fit using the selected features, and the result is returned.

    Parameters
    ----------
    data : pandas.DataFrame
           A dataframe containing the features and the response variable.
    response : str
               The name of the response variable in the data.
    neg_coeff_columns : list of str
                        A list of column names for features that should have negative coefficients in the final model.
    necessary_col : list of str
                    A list of column names for features that should be included in the model at the start of the selection process.
    correlation_data : pandas.DataFrame
                       A correlation matrix of the features in the data."

    Returns
    -------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
            The result of fitting a linear regression model on the selected features.
    """
    remaining = set(data.columns)
    remaining.remove(response)
    # remaining.remove()

    """Starting with necessary columns"""
    selected = necessary_col
    best_new_score = 0  # p value

    while remaining and best_new_score <= 0.1:
        scores_with_candidates = []
        model_coeff = []
        for candidate in remaining:
            # print(f'loop {candidate}')
            formula = "{} ~ {} + 1".format(response, " + ".join(selected + [candidate]))
            score = smf.ols(formula, data).fit().pvalues.reset_index().iloc[-1, -1]
            # print(f'printing score:{score}')
            score_all = (
                smf.ols(formula, data).fit().pvalues.reset_index().iloc[1:, 1].tolist()
            )
            # print(f'printing score_all:{score_all}')

            """Coeff-check"""
            coeff_table = coefficient_check(
                formula, data, neg_coeff_columns, response, correlation_data
            )

            scores_with_candidates.append((score, candidate, score_all, coeff_table))
            model_coeff.append(coeff_table)

        scores_with_candidates.sort(reverse=True)

        """Keeping only combinations with significant p values"""
        score_with_candidates_updated = [
            (i[0], i[1], i[2], i[3])
            for i in scores_with_candidates
            if all(j <= 0.1 for j in i[2])
        ]
        score_with_candidates_updated.sort(reverse=True)

        """Keeping only combinations with intuitive signs"""
        score_with_candidates_updated = [
            (j[0], j[1], j[2], j[3])
            for j in score_with_candidates_updated
            if all(
                (np.sign(j[3]["coef"]) == j[3]["sign"])
                & (np.sign(j[3]["coef"]) == np.sign(j[3]["corr_sign"]))
            )
        ]
        score_with_candidates_updated.sort(reverse=True)

        if len(score_with_candidates_updated) == 0:
            print("No remaining iterations where log_total_visits is significant")
            break

        # print(score_with_candidates_updated[-1])
        (
            best_new_score,
            best_candidate,
            all_scores,
            table_coeff,
        ) = score_with_candidates_updated[-1]
        # if len(all_scores)==0:
        # all_scores.append(0.1)
        # print(best_new_score,all_scores)
        if all(i <= 0.1 for i in all_scores):
            remaining.remove(best_candidate)
            selected.append(best_candidate)
        elif all_scores[0] > 0.1:
            print(
                "log_total_visits is insignificant. "
                + "pvalue is: "
                + str(all_scores[0])
            )
            break

    formula = "{} ~ {} + 1".format(response, " + ".join(selected))
    model = smf.ols(formula, data).fit()

    return model


def get_results(
    train_X,
    train_y,
    label_sales,
    tune_on,
    reg_vars,
    rtm_vars,
    cost_col,
    cost_mac,
    log_visits,
):
    """Compute various regression metrics and rtm contribution for a given set of features.

    The function performs a regression using the specified features and target variable, and computes various regression metrics such as R-squared, adjusted R-squared, MAPE, and weighted MAPE. It also computes the contribution of each feature to the regression and the ROI of the RSV and MAC channels.
    The result is returned as a dictionary.

    Parameters
    ----------
    train_X : pandas.DataFrame
              A dataframe containing the features used for the regression.
    train_y : pandas.Series
              A series containing the target variable for the regression.
    label_sales : str
                  A label for the sales data being analyzed.
    tune_on : list of str
              A list of column names for features that should be included in the model at the start of the selection process.
    reg_vars : list of str
               A list of column names for features that should be used in the regression model.
    rtm_vars : list of str
               A list of column names for features that correspond to the RTM channels.
    cost_col : str
               The name of the column containing the total visit cost.
    cost_mac : str
               The name of the column containing the RSV-MAC after conversion.
    log_visits : str
                 The name of the column containing the log of total visits.

    Returns
    -------
    iter_result : dict
                  A dictionary containing various regression metrics and rtm contribution for the given set of features.
    """

    iter_result = dict()
    iter_result.update({"added_vars": reg_vars.copy()})

    for variable in rtm_vars:
        if variable not in reg_vars:
            reg_vars.append(variable)

    reg_vars.extend(tune_on)

    reg_vars = list(set(reg_vars))
    print("Regression Variables :", reg_vars)

    rtm_cols = list(set.intersection(set(rtm_vars), set(reg_vars)))

    # outlier detection

    fil = train_y.values.ravel() < np.percentile(
        train_y.values.ravel(), 95
    )  # Removing top 5% of stores based on Sales??

    # regression pipeline
    reg_ppln_ols = Pipeline(
        [
            (
                "",
                FunctionTransformer(
                    _custom_data_transform, kw_args={"cols2keep": reg_vars}
                ),
            ),
            ("estimator", SKLStatsmodelOLS()),
        ]
    )

    reg_ppln_ols.fit(
        train_X.loc[fil, train_X.columns[train_X.columns.isin(reg_vars)]],
        train_y[fil].values.ravel(),
    )

    """OLS estimator"""
    ols_estimator = reg_ppln_ols["estimator"]
    coeff_dict = ols_estimator.coeff_table().to_dict()["coef"]
    coeff_dict_full = coeff_dict.copy()
    coeff_dict.pop("intercept")

    """MAPE & Weighted MAPE"""
    coeff_smry = ols_estimator.coeff_table()
    Rsqd = ols_estimator.model_.rsquared
    adjRsqd = ols_estimator.model_.rsquared_adj
    y_hat = reg_ppln_ols.predict(train_X.loc[fil, reg_vars])
    y = train_y[fil].values
    y = y.ravel()
    MAPE = np.mean(np.ma.masked_invalid(np.abs(y - y_hat) / y))
    wt_MAPE = np.mean(np.abs(y - y_hat)) / np.mean(y)

    """Appending Cluster label & Number of stores"""
    iter_result.update({"Label_sales": label_sales})
    iter_result.update({"No. of Records": sum(fil)})

    """Contribution of variables in the model"""
    contrib, rtm_contrib = get_contribution(
        train_X[fil], train_y[fil], coeff_dict, rtm_cols
    )

    """RSV ROI & MAC ROI"""
    RSV_ROI = get_ROI(train_X[fil], train_y[fil], coeff_dict, rtm_cols, cost_col)
    MAC_ROI = get_ROI(train_X[fil], train_y[fil], coeff_dict, rtm_cols, cost_mac)

    """Updating RTM contribution, ROI, R2, MAPE, Features"""
    iter_result.update({"Rtm_cont_sum": np.sum([val for val in rtm_contrib.values()])})
    iter_result.update({"RSV_ROI": RSV_ROI})
    iter_result.update({"MAC_ROI": MAC_ROI})
    iter_result.update({"R2_score": Rsqd})
    iter_result.update({"Adj_r2_score": adjRsqd})
    iter_result.update({"MAPE": MAPE})
    iter_result.update({"Weighted_mape": wt_MAPE})
    iter_result.update({"Features": reg_vars})
    iter_result.update({"N_features": len(reg_vars)})

    """Updating coefficients & contributions"""
    iter_result.update(coeff_dict_full)
    for i in list(contrib.keys()):
        contrib[i + "_contrib"] = contrib.pop(i)
    iter_result.update(contrib)

    """vif for each coefficient"""
    vif_dict = ols_estimator.coeff_table(add_vif=True).to_dict()["VIF"]
    for i in list(vif_dict.keys()):
        vif_dict[i + "_vif"] = vif_dict.pop(i)
    iter_result.update(vif_dict)

    """P values"""
    p_val = dict(coeff_smry["P>|t|"])
    for i in list(p_val.keys()):
        p_val[i + "_pvalue"] = p_val.pop(i)
    # iter_result.update(p_val)
    plot_res = all(p < 0.2 for p in iter(p_val.values())) and (  # noqa
        coeff_dict[log_visits] > 0
    )
    # plot_res = all(p < .1 for p in iter(p_val.values())) and (coeff_dict['total_visits'] > 0)
    # for i in reg_vars:
    #     if 'product_count' in i:
    #         plot_res = plot_res and (coeff_dict[i] > 0)
    # plot_res=True
    # iter_result.update(p_val)
    if True:
        plot_res1 = any(p > 0.1 for p in iter(p_val.values()))
        iter_result.update({"pval>.1": plot_res1})
        iter_result.update(p_val)
        return iter_result


# rtm related functions
def get_ROI(train_X, train_y, coeff_dict, rtm_cols, cost_col):
    """Calculate the RSV ROI.

    The function calculates the contribution of RTM, based on the model coefficients and the training data, and then calculates the RSV ROI using the contribution of the RTM and the total visit cost. The result is returned as a float.

    Parameters
    ----------
    train_X : pandas.DataFrame
              A dataframe containing the features used for the regression.
    train_y : pandas.DataFrame
              A dataframe containing the target variable for the regression.
    coeff_dict : dict
                 A dictionary containing the coefficients for the features used in the regression.
    rtm_cols : list of str
               A list of column names for features that correspond to the RTM activities.
    cost_col : str
               The name of the column containing the total visit cost."

    Returns
    -------
    ROI: float
         (ROI in terms of RSV)
    """
    _, rtm_contrib = get_contribution(train_X, train_y, coeff_dict, rtm_cols)
    rtm_impact = sum(rtm_contrib.values()) * train_y.sum()
    cost = train_X[cost_col].sum()
    return rtm_impact / cost


# Add optimal visits
def get_optimal_visits(
    coeff_dict, visit_var, visits_transformation, RSV_TO_MAC, cost_per_visit
):
    """Calculate the optimal number of visits for RTM, given the model coefficients and other parameters.

    The function first calculate the RTM contribution for each possible number of visits, using the model coefficients and the specified transformation function. It then calculates the ROI and incremental ROI for each number of visits, using the RSV_TO_MAC conversion and the cost per visit. Finally, it plots the RTM contribution, ROI, and incremental ROI for each number of visits, and returns the optimal number of visits and the plot object as a tuple.

    Parameters
    ----------
    coeff_dict : dict
                 A dictionary containing the coefficients for the features used in the regression.
    visit_var : str
                The name of the feature referring to the RTM visits in the dataframe.
    visits_transformation : callable
                            A function that takes the number of visits as input and returns a transformed value.
    RSV_TO_MAC : float
                 A finance metric used to convert RSV to MAC.
    cost_per_visit : float
                     The calculated cost of a single visit.

    Returns
    -------
    (optimal_visits,ax1): tuple
                    A tuple containing the optimal number of visits and the plot object used to visualize the results.
    """
    visits_coeff = coeff_dict.get(visit_var)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    max_visits = 12
    inc_roi = []
    # Only varying factor in the hypothetical scenario is total number visits.
    # Hence, only total visits contrib is reqd to calculate optimal visits
    # Optimal visits is when incremental because of additional 1 visits is less than the cost of a single visit.
    while not any([inc_ < 1 for inc_ in inc_roi]):
        visits = range(1, max_visits + 1)
        contrib = [visits_transformation(nvisit) * visits_coeff for nvisit in visits]
        roi = [
            c * RSV_TO_MAC / (cost_per_visit * nvisit)
            for nvisit, c in zip(visits, contrib)
        ]
        inc_roi = [
            (c_ - c) * RSV_TO_MAC / cost_per_visit
            for c, c_ in zip(contrib[:-1], contrib[1:])
        ]
        max_visits += 5

    optimal_visits = [i for i, inc_ in enumerate(inc_roi) if inc_ < 1][0]
    line1 = ax1.plot(visits, contrib, color="b", label="RTM contribution")
    line2 = ax2.plot(visits, roi, color="m", label="ROI")
    line3 = ax2.plot(visits[1:], inc_roi, color="r", label="Incremental ROI")
    ax2.plot(visits, [1] * len(visits), "r--")
    ax1.set_xlabel("nVisits")
    ax1.set_ylabel("Contribution to RSV")
    ax2.set_ylabel("ROI")
    ax1.grid(False)
    lns = line1 + line2 + line3
    ax1.legend(lns, [line.get_label() for line in lns], loc=0)
    return (optimal_visits, ax1)


def get_contribution(train_X, train_y, coeff_dict, rtm_cols):
    """Contribution-calculating function based on model coefficients and train data.

    This function calculates the percentage contribution of each feature and the RTM features by multiplying the feature's coefficient with the sum of that feature's values in train_X, and then dividing the result by the sum of the train_y values. The output is provided as two dictionaries: contrib_dict contains the percentage contribution of all features in the model, and rtm_contrib contains the percentage contribution of only the RTM features.

    Parameters
    ----------
    train_X : pd.DataFrame
              Training input dataset, where each row represents an instance and each column represents a feature.
    train_y : pd.DataFrame
              Training output dataset, where each row represents the output of an instance.
    coeff_dict : dict
                 Dictionary of features (same as column names in train_X) and the coefficients.
    rtm_cols : list of str
               List of column names describing RTM .

    Returns
    -------
    contrib_dict : dict
                   Dictionary of the percentage contribution of all features in the model.
    rtm_contrib : dict
                  Dictionary of the percentage contribution for RTM features."
    """
    contrib = (
        train_X[coeff_dict.keys()].sum(0).multiply(pd.Series(coeff_dict))
        / train_y.sum()
    )
    contrib_dict = contrib.to_dict()
    rtm_contrib = {c: contrib_dict.get(c, np.NaN) for c in rtm_cols}
    return contrib_dict, rtm_contrib


# Custom Transformations like these can be utilised
def _custom_data_transform(df, cols2keep=None):
    """Customised Transformer to eliminate some data columns.

    This function takes a DataFrame and an optional list of columns to keep. If the list of columns to keep is provided, the function returns a DataFrame containing only those columns. If the list is not provided or is empty, the function returns the original DataFrame unchanged.

    Parameters
    ----------
    df : pd.DataFrame
         Input DataFrame to be transformed.
    cols2keep : list of str, optional
                List of column names to keep in the DataFrame. If not provided or empty, all columns are kept. (default: None)"

    Returns
    -------
    df : pd.DataFrame
        Transformed DataFrame containing only the specified columns or the original DataFrame if cols2keep is not provided or empty.
    """
    cols2keep = cols2keep or []
    if len(cols2keep):
        return df.select_columns(cols2keep)
    else:
        return df


def drop_cols(df, contains_keyword: str, except_cols=None, case_sensitive=False):
    """Drop columns from a DataFrame based on a keyword contained in their names.

    This function takes a DataFrame and drops columns whose names contain the specified keyword. It provides an optional list of exception columns that should not be dropped even if their names contain the keyword. The case sensitivity of the keyword matching can also be specified.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame from which columns need to be dropped.
    contains_keyword : str
                       The keyword to search for in column names.
    except_cols : list of str, optional
                  A list of column names that should not be dropped even if their names contain the keyword. (default: None)
    case_sensitive : bool, optional
                     If True, the keyword matching will be case-sensitive. If False, the keyword matching will be case-insensitive. (default: False)

    Returns
    -------
    cols : pd.DataFrame
           A DataFrame with columns dropped if their names contain the specified keyword.

    """
    cols = list(
        df.columns[df.columns.str.contains(contains_keyword, case=case_sensitive)]
    )
    if except_cols is not None:
        cols = list(filter(lambda i: i not in except_cols, cols))
    print(cols)
    return cols


# This is the standard figure size used in the notebooks
figure_size = (8, 6)


# Line plots


def line_plot(numerical_series, xlabel: str, ylabel: str):
    """Plot a line graph from a series of continuous numerical data.

    Parameters
    ----------
    numerical_series: numpy.ndarray or pd.Series
                      A 1-D array or series with continuous numerical data.
    xlabel: str
            The label for the x-axis of the plot.
    ylabel: str
            The label for the y-axis of the plot.
    figure_size: tuple, optional (default=(8, 6))
                 The figure size of the plot in inches (width, height).
    """

    plt.figure(figsize=figure_size)
    sns.lineplot(data=numerical_series)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(xlabel + " vs " + ylabel)
    plt.show()


# distribution plot
def distribution_plot(series_data, series_title: str):
    """Plot a distribution plot to show how the series data is spread.

    Parameters
    ----------
    series_data: numpy.ndarray or pd.Series
                 A 1-D array or series with numerical data.
    series_title: str
                  The title for the distribution plot.
    figure_size: tuple, optional (default=(8, 6))
                 The figure size of the plot in inches (width, height).
    """
    if np.nanvar(series_data) != 0:
        plt.figure(figsize=figure_size)
        sns.distplot(series_data, kde=True)
        plt.title("Distribution of " + series_title)
        plt.show()


# count plot
def count_plot(title: str, x_axis: str, data, hue=None, *args, **kwargs):
    """Plot a bar graph with different labels.

    Parameters
    ----------
    title: str
        The title for the plot.
    x_axis: str
        The column name for the x-axis of the plot.
    data: pd.DataFrame
        The dataframe containing the data.
    hue: str, optional (default=None)
        The variation of each aspect.
    figure_size: tuple, optional (default=(8, 6))
        The figure size of the plot in inches (width, height).
    """
    plt.figure(figsize=figure_size)
    sns.countplot(x=x_axis, hue=hue, data=data, palette="rainbow")
    plt.title(title)
    plt.xticks(rotation=45)

    plt.show()


# correlation plot
def correlation(dataframe):
    """Plot a correlation matrix heatmap for a dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe containing the data.
    figure_size: tuple, optional (default=(20, 12))
        The figure size of the plot in inches (width, height).
    """
    plt.figure(figsize=(20, 12))
    sns.heatmap(
        dataframe.corr(), mask=mask(dataframe), annot=True, annot_kws={"size": 14}
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.set_style("white")


def mask(dataframe):
    """Create a mask for the correlation plot to hide the upper triangle.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe containing the data.

    Returns
    -------
    numpy.ndarray
    A 2-D boolean array for masking the upper triangle of the correlation plot.
    """
    mask = np.zeros_like(dataframe.corr())
    triangle_indices = np.triu_indices_from(mask)
    mask[triangle_indices] = True


def joint_plot(df, x: str, y: str, title: str):
    """Plot a joint plot of two variables.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the data.
    x: str
       The column name for the x-axis of the plot.
    y: str
       The column name for the y-axis of the plot.
    title: str
           The title for the plot.
    height: int, optional (default=8)
            The height of the plot in inches."
    """
    sns.set(font_scale=1.5)
    sns.jointplot(data=df, x=x, y=y, height=8, kind="reg")
    plt.title(title)
    plt.show()


def lm_plot(df, x: str, y: str, col: str):
    """Plot a lm plot of two variables.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the data.
    x: str
        The column name for the x-axis of the plot.
    y: str
        The column name for the y-axis of the plot.
    col: str
        The column name for the column facet of the plot.
    figure_size: tuple, optional (default=(20, 12))
        The figure size of the plot in inches (width, height).
    """
    plt.figure(figsize=(20, 12))
    sns.set(font_scale=1.5)
    sns.lmplot(data=df, x=x, y=y, aspect=1, height=8, col=col)
    plt.show()


def categorical_plots(df, x: str, y: str, hue=None):
    """Plot a categorical plot of two variables using violin plot and swarm plot.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the data.
    x: str
        The column name for the x-axis of the plot.
    y: str
        The column name for the y-axis of the plot.
    hue: str, optional (default=None)
        The column name for the hue variable of the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.5)
    sns.violinplot(data=df, x=x, y=y, hue=hue, dodge=True)
    sns.swarmplot(data=df, x=x, y=y, hue=hue, dodge=True, color="black")
    plt.show()


def scatter_plot(df, xcol: str, ycol: str):
    """Plot a scatter plot of two variables.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the data.
    xcol: str
        The column name for the x-axis of the plot.
    ycol: str
        The column name for the y-axis of the plot.
    """

    plt.scatter(x=df[xcol], y=df[ycol], c=df[ycol], alpha=0.3, cmap="Spectral")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(ycol + " vs " + xcol)
    plt.colorbar()


def boxplots_for_bannerlevel(y: str, df):
    """Plot a boxplot for each banner level.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the data.
    y: str
        The column name for the y-axis of the plot.
    """
    plt.figure(figsize=(20, 12))

    sns.boxplot(x="Banner", y=y, data=df)
    plt.xlabel("Banner", fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.title("Boxplot of " + y)

    plt.show()


def getting_correlated_columns(dataframe, thresold):
    """Get a set of column names with correlation above a given threshold.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe containing the data.
    threshold: float
        The threshold value for correlation. The function returns a set of
        column names with correlation above this threshold.

    Returns
    -------
    correlated_columns : set
                         A set of column names with correlation above the given threshold.
    """

    correlated_columns = set()
    corr_matrix = dataframe.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > thresold:
                correlated_columns.add(corr_matrix.columns[i])
    return correlated_columns


def ads_segregation(
    ads_dataframe,
    get_epos=True,
    sales_value_column="sales_value",
    identity_cols=None,
    cat_cols=None,
):
    """Function is used to divide the input ADS DataFrame based on whether it contains EPOS data or not.   # noqa

    If `get_epos` is True, the output will contain EPOS data, otherwise it will contain non-EPOS data. For EPOS data, the output will also include numerical and categorical columns, as well as identity columns.

    Parameters
    ----------
    ads_dataframe: pd.DataFrame
        The ADS DataFrame to be segregated.
    get_epos: bool, optional (default=True)
        Whether to segregate the EPOS data or the non-EPOS data.
    sales_value_column: str, optional (default='sales_value')
        The name of the column that contains sales values.
    identity_cols: list, optional (default=None)
        A list of column names used to identify the stores.
    cat_cols: list, optional (default=None)
        A list of column names that are categorical in nature.

    Returns
    -------
    tuple
    A tuple of DataFrames based on the segregation performed.
    The tuple consists of:
    - data_epos if get_epos is True, else data_no_epos
    - data_numeric_epos if get_epos is True, else data_numeric_no_epos
    - data_identity_epos if get_epos is True, else data_identity_no_epos
    - data_cat_epos if get_epos is True, else data_cat_no_epos
    """
    if get_epos is True:
        data_epos = ads_dataframe.loc[
            ~ads_dataframe.loc[:, sales_value_column].isna()
        ].reset_index(drop=True)
        data_numeric_epos = data_epos.select_dtypes(include=np.number).copy()
        data_identity_epos = data_epos.loc[:, identity_cols].copy()
        data_cat_epos = pd.DataFrame(
            data_epos.loc[:, cat_cols].copy(), dtype="category"
        )
        print(data_epos.shape)
        return data_epos, data_numeric_epos, data_identity_epos, data_cat_epos
    else:
        data_no_epos = ads_dataframe.loc[
            ads_dataframe.loc[:, sales_value_column].isna()
        ].reset_index(drop=True)
        data_numeric_no_epos = data_no_epos.select_dtypes(exclude=np.number).copy()
        data_identity_no_epos = data_no_epos.loc[:, identity_cols].copy()
        data_cat_no_epos = pd.DataFrame(
            data_no_epos.loc[:, cat_cols].copy(), dtype="category"
        )
        print(data_no_epos.shape)
        return (
            data_no_epos,
            data_numeric_no_epos,
            data_identity_no_epos,
            data_cat_no_epos,
        )


def drop_constant_value_cols(df, numeric_data=True):
    """Drop columns from a DataFrame that have constant values.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to remove constant value columns from.

    numeric_data : bool, optional (default=True)
        Whether to only consider numerical data when searching for constant value columns.

    Returns
    -------
    df : pandas.DataFrame
         The DataFrame with constant value columns removed.
    """
    if numeric_data is True:
        constant_fields = list(df.var()[df.var() == 0].index)

        df.drop(labels=constant_fields, axis=1, inplace=True)

        return df
    else:
        return df[df.columns[df.nunique(dropna=True) > 1]]


def create_missing_value_df(df):
    """Create a DataFrame that shows the proportion of missing values for each column in a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to create the missing value DataFrame from.

    Returns
    -------
    missing_value_df : pandas.DataFrame
            A DataFrame with column names of df as index and the proportion of missing values for each column.

    """
    missing_value_df = (
        df.isna().sum().sort_values(ascending=False) / df.shape[0]
    ).to_frame()
    missing_value_df.reset_index(inplace=True)
    missing_value_df.columns = list(["fields", "proportion"])
    return missing_value_df


def find_missing_value_columns(df, thres=0.2):
    """Find the columns in a DataFrame that have missing value proportions greater than a threshold.

    Parameters
    ----------
    df : pandas.DataFrame
         The DataFrame to find the columns with missing values in.
    thres : float, optional (default=0.2)
            The threshold to use for finding columns with missing values.

    Returns
    -------
    list
    A list of the columns with missing value percentage greater than the threshold.
    """

    missing_value_df = create_missing_value_df(df)
    print(f"Number of fields with missing value proportion in epos data > {thres} are:")
    print(missing_value_df.loc[missing_value_df.proportion > thres, "fields"].count())
    return list(
        missing_value_df.loc[missing_value_df.proportion > thres, "fields"].values
    )


def imputation_with_thresold(df, thres=0.2, strategy="median"):
    """Impute missing values in a DataFrame by either dropping columns exceeding a threshold or filling them with a strategy.

    Parameters
    ----------
    df : pandas.DataFrame
            The DataFrame to impute missing values in.
    thres : float, optional (default=0.2)
            The threshold to use for dropping columns with missing values.
    strategy : str, optional (default='median')
            The imputation strategy to use for filling missing values.
            Allowed values are ['median','mean','constant'].

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame after dropping columns exceeding the threshold and filling missing values.
    """

    # dropping the columns with mising value proportion greater than thresold
    cols = find_missing_value_columns(df, thres)
    df.drop(labels=cols, axis=1, inplace=True)
    # Now, we are left with columns containing missing value proportion less than or equals to thresold

    # start imputing
    imputer = SimpleImputer(strategy=strategy)

    for col in df.columns:
        df.loc[:, col] = imputer.fit_transform(df[col].values.reshape(-1, 1))[:, 0]
    return df


def eliminate_highly_correlated_columns(df, thres=0.75):
    """Remove columns from a DataFrame that are highly correlated with each other.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to remove highly correlated columns from.

    thres : float, optional (default=0.75)
            The threshold to use for determining which columns are highly correlated.

    Returns
    -------
    df : pandas.DataFrame
         The DataFrame after removing highly correlated columns."
    """
    df = df.copy()
    corr_matrix = df.corr().abs()
    corr_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(corr_mask)
    to_drop = [col for col in df.columns if any(tri_df[col] > thres)]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def classification_metrics(actual, predicted):
    """Calculate the classification metrics for a given set of actual and predicted values.

    Parameters
    ----------
    actual : array-like of shape (n_samples,)
            The actual values of the dependent variable.

    predicted : array-like of shape (n_samples,)
            The predicted values of the dependent variable.

    Returns
    -------
    tuple
    A tuple containing the confusion matrix, classification report and accuracy of the prediction.
    """
    cm = confusion_matrix(actual, predicted)
    print("Confusion Matrix:")
    print(cm)
    cr = classification_report(actual, predicted)
    print(
        "Classification Report:",
    )
    print(cr)
    accuracy = accuracy_score(actual, predicted)
    print("Accuracy:", accuracy)
    return cm, cr, accuracy
