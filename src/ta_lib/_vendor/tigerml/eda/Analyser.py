from .base import EDAReport
from .segmented import SegmentedEDAReport
from .time_series import SegmentedTSReport, TSReport


class Analyser:
    """Analyser class."""

    # FIXME: Timestamps should not be treated as categorical variables

    def __new__(cls, data, segment_by=None, *args, **kwargs):
        """Returns eda report."""
        if segment_by is not None:
            return SegmentedEDAReport(data, segment_by, *args, **kwargs)
        else:
            return EDAReport(data, *args, **kwargs)


class TSAnalyser:
    """Ts Analyser class."""

    def __new__(cls, data, ts_column, ts_identifiers=None, *args, **kwargs):
        """Returns TS report."""
        if ts_identifiers is not None:
            return SegmentedTSReport(data, ts_column, ts_identifiers, *args, **kwargs)
        else:
            return TSReport(data, ts_column, *args, **kwargs)

    # def __init__(self, data, ts_column, ts_identifiers=None, y=None, y_continuous=None):
    #     self.ts_column = ts_column
    #     self.ts_identifiers = ts_identifiers
    #     super().__init__(data, segment_by=ts_identifiers, y=y, y_continuous=y_continuous)
    #     # self.data.set_index(self.ts_column, inplace=True)
    #     self.data.sort_values(by=[self.ts_column], inplace=True)
    #
    # def get_report(self, y=None, quick=True, *args, **kwargs):
    #     super().create_report(y=y, quick=quick)
    #     self.report['time_series_analysis'] = self.get_time_series_analysis()
    #     super().save_report(*args, **kwargs)
