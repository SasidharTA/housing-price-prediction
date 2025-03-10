from ...contents import Table


class HTMLTable(Table):
    """Html table class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Html table class."""
        return cls(parent.styler, parent.name, parent.datatable)

    def to_html(self, resource_path=""):
        """Converts to html for Html table class."""
        # html_str = ''
        # if self.name:
        #     html_str += title_html(prettify_slug(self.name))
        if self.conditional_formatters:
            for rule in self.conditional_formatters:
                if rule["cols"]:
                    if rule["col_wise"]:
                        if not isinstance(rule["cols"], list):
                            rule["cols"] = [rule["cols"]]
                        for col in rule["cols"]:
                            self.styler.background_gradient(
                                subset=[col], cmap=rule["style"], axis=0
                            )
                    else:
                        self.styler.background_gradient(
                            subset=rule["cols"], cmap=rule["style"], axis=0
                        )
                elif rule["rows"]:
                    self.styler.background_gradient(
                        subset=rule["rows"], cmap=rule["style"], axis=1
                    )
        html_str = '<div class="content_inner">{}</div>'.format(self.styler.to_html())
        return '<div class="content table_content {}">{}</div>'.format(
            "apply_datatable" if self.datatable else "no_datatable", html_str
        )

    def save(self, name=""):
        """Saves for Html table class."""
        html_str = "<html><body>{}</body></html>".format(self.to_html())
        if not name:
            name = self.name
        name = name + (".html" if name[:-5] != ".html" else "")
        f = open(name, "w")
        f.write(html_str)
        f.close()
