import gc
import logging
import os
import tigerml.core.dataframe as td
from tigerml.core.preprocessing import DataProcessor
from tigerml.core.reports import create_report, format_tables_in_report
from tigerml.core.utils import compute_if_dask, measure_time, time_now_readable
from tigerml.core.utils.constants import (
    MIN_CUTOFF_FOR_KEY_HEURISTIC,
    NA_VALUES,
    SUMMARY_KEY_MAP,
)

from .plotters import (
    FeatureAnalysisMixin,
    FeatureInteractionsMixin,
    HealthMixin,
    KeyDriversMixin,
)

_LOGGER = logging.getLogger(__name__)


class EDAReport(
    DataProcessor,
    HealthMixin,
    FeatureAnalysisMixin,
    FeatureInteractionsMixin,
    KeyDriversMixin,
):
    """EDA toolkit for classification and regression models.

    To evaluate and generate reports to summarize, data health, univariate & bivariate analyis, interactions and keydrivers.

    Parameters
    ----------
    data : pd.DataFrame, dataframe to be analyzed

    y : string, default=None
        Name of the target column

    y_continuous : bool, default=None
        Set to False, for classificaiton target

    Examples
    --------
    >>> from tigerml.eda import EDAReport
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> an = EDAReport(df, y = 'Survived', y_continuous = False)
    >>> an.get_report(quick = True)
    """

    def __init__(self, data, y=None, y_continuous=None):
        if not data.__module__.startswith("tigerml"):
            data = td.DataFrame(data)
            data = data.convert_datetimes()
        super().__init__(data, y=y, y_continuous=y_continuous)

    def _create_report(self, y=None, quick=True, corr_threshold=None):
        if y:
            self._set_xy_cols(y)
        self.report = {}
        # import pdb
        # pdb.set_trace()
        self.report["data_preview"] = {
            "head": [self.data.head(5)],
            "tail": [self.data.tail(5)],
        }
        self.report["health_analysis"] = self.health_analysis()
        self.report["data_preview"]["pre_processing"] = self._prepare_data(
            corr_threshold
        )
        self.report["feature_analysis"] = self.feature_analysis()
        self.report["feature_interactions"] = self.feature_interactions()
        if self.y_cols:
            self.report["key_drivers"] = self.key_drivers(quick=quick, y=self.y_cols)
        else:
            _LOGGER.info(
                "Could not generate key drivers report as dependent variable is not defined"
            )

    def _save_report(
        self, format=".html", name="", save_path="", tiger_template=False, **kwargs
    ):
        if not name:
            name = "data_exploration_report_at_{}".format(time_now_readable())
        compute_if_dask(self.report)
        create_report(
            self.report,
            name=name,
            path=save_path,
            format=format,
            split_sheets=True,
            tiger_template=tiger_template,
            **kwargs
        )
        del self.report
        gc.collect()

    def get_report(
        self,
        format=".html",
        name="",
        y=None,
        corr_threshold=None,
        quick=True,
        save_path="",
        tiger_template=False,
        light_format=True,
        **kwargs
    ):
        """Create consolidated report on data preview,feature analysis,feature interaction and health analysis.

        The consolidated report also includes key driver report if y(target dataframe) is passed while
        calling create_report.

        Parameters
        ----------
        y : str, default = None
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        name : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        quick : boolean, default=True
            If true,calculate SHAP values and create bivariate plots
        corr_threshold : float, default=None
            To specify correlation threshold
        excel_params : dict
            Dictionary containing the following keys if the format is ".xlsx".
            If a key is not provided, it will take the default values.

                - have_plot : boolean; default False.
                    If True, keep the plots in image format in excel report.
                - n_rows : int; default 100.
                    Number of sample rows to keep for plot types containing all the records in data (for example, density plot, scatter plot etc.)
        """
        self._create_report(y=y, quick=quick, corr_threshold=corr_threshold)
        if light_format:
            self.report = format_tables_in_report(self.report)

        if format == ".xlsx":
            keys_to_combine = [
                ("data_preview", "pre_processing", "encoded_mappings"),  # noqa
                ("feature_analysis", "distributions", "numeric_variables"),  # noqa
                ("feature_analysis", "distributions", "non_numeric_variables"),  # noqa
                (
                    "feature_interactions",
                    "bivariate_plots (Top 50 Correlations)",
                ),  # noqa
                ("key_drivers", self.y_cols[0], "bivariate_plots"),
            ]  # noqa

            from tigerml.core.utils import convert_to_tuples

            convert_to_tuples(keys_to_combine, self.report)

        return self._save_report(
            format=format,
            name=name,
            save_path=save_path,
            tiger_template=tiger_template,
            **kwargs
        )

    def _get_ireport(self, y=None, port=5006, save=True):
        # TBD: Is this applicable now?
        try:
            from tigerml.viz import DataExplorer
        except ImportError as e:
            raise ImportError("interactive report requires tigerml.viz package.", e)
        import panel as pn

        pn.extension()
        health_analysis = self.health_analysis()
        preview = {"head": [self.data.head(5)], "tail": [self.data.tail(5)]}
        summaries = {
            "numeric_summary": [self.numeric_summary()],
            "non_numeric_summary": [self.non_numeric_summary()],
        }
        interactions = {
            "correlation_table": [self.correlation_table()],
            "correlation_matrix": [self.correlation_heatmap()],
        }
        report_content = {
            "data_preview": preview,
            "health_analysis": health_analysis,
            "summary_stats": summaries,
            "interactions": interactions,
        }
        if y:
            self._set_xy_cols(y)
        if self._current_y:
            interactions["feature_correlation_with_{}".format(self._current_y)] = [
                self.get_feature_scores(quick=True)["feature_correlation"]
            ]
        from tigerml.core.reports.html import create_html_report

        report = create_html_report(report_content, save=False)
        if save:
            report.save()
        report_content.update(
            {
                "explorer": {
                    "widget": ['<div id="de_widget">{{ embed(roots.de_widget) }}</div>']
                }
            }
        )
        report = create_html_report(report_content, save=False)
        _thisdir = os.path.split(__file__)[0]
        style_file = open(
            os.path.join(_thisdir, "../reports/html/report_resources", "style.css"), "r"
        )
        custom_style = style_file.read()
        style_file.close()
        template = """
            {% extends base %}
            {% block postamble %}
            <style>
                {{custom_style}}
                #widget .content_inner {
                    display: block;
                }
            </style>
            <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.css">
            <script src="https://code.jquery.com/jquery-3.4.1.min.js"
            integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=" crossorigin="anonymous"></script>
            <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.js"></script>
            <script>
                $(document).ready( function () {
                    datatables = $('.apply_datatable table').DataTable();
                    $('.side_nav li.group > a').click(function() {
                        $(this).toggleClass('closed');
                    });
                });
            </script>
            {% endblock %}
            {% block contents %}
            <div class="side_nav"><ul class="side_nav_inner">{{report_nav}}</ul></div>
            <div class="report_body">{{report_body}}</div>
            {% endblock %}
        """
        template = template.replace("{{report_nav}}", report._get_nav())
        template = template.replace("{{report_body}}", report.to_html())
        template = template.replace("{{custom_style}}", custom_style)
        tmpl = pn.Template(template)
        de = DataExplorer(self.data)
        de._initiate()
        tmpl.add_panel("de_widget", de.pane)
        # self.pane = pn.Column(health_analysis, summaries, interactions, de.pane)
        tmpl.show(port)
        de.update_plot(bokeh=True)
