import pandas as pd


class DataSet:
    """Dataset class."""

    def __init__(self, data, fetch=None, groupby=None, filter=None):
        if not isinstance(data, pd.DataFrame):
            raise Exception("data can only be a pandas DataFrame")
        self.data = data
        self._fetch = fetch
        self._groupby = groupby
        self._filter = filter

    def get_fetch(self):
        """Returns fetch attribute."""
        return self._fetch

    def get_groupby(self):
        """Returns groupby attribute."""
        return self._groupby

    def get_filter(self):
        """Returns filter attribute."""
        return self._filter

    def _validate_fetch(self, fetch_cols):
        """Returns fetch cols attribute."""
        return fetch_cols

    def _validate_groupby(self, groupby_cols):
        """Returns groupby cols attribute."""
        return groupby_cols

    def _validate_filter(self, filter_rules):
        """Returns filter rules attribute."""
        return filter_rules

    def fetch(self, fetch):
        """Initializes fetch attribute."""
        self._fetch = self._validate_fetch(fetch)
        return self

    def groupby(self, groupby):
        """Initializes groupby attribute."""
        self._fetch = self._validate_groupby(groupby)
        return self

    def filter(self, filter):
        """Initializes filter attribute."""
        self._fetch = self._validate_filter(filter)
        return self

    def run(self):
        """Initializes fetch attribute."""
        return self.data[self.filter].groupby(by=self.groupby)[self.fetch]
