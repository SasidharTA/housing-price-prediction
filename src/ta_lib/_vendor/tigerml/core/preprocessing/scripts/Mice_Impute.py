import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression


class MiceImputer:

    model_dict_ = {}

    def __init__(self, seed_strategy="mean", target=None, group=[]):
        # self.seed_nulls = seed_nulls
        self.seed_strategy = seed_strategy
        self.target = target
        self.group = group

    def transform(self, X):
        if not self.group:
            self.group = list(X.columns)
        key = X.index.name
        y = X[self.target]
        x_null = X.loc[y.isnull(), self.group]
        y_null = y[y.isnull()].reset_index()[key]
        y_notnull = y[y.notnull()]

        x_null = self.imputer.transform(x_null)

        pred = pd.concat(
            [
                pd.Series(self.model.predict(x_null)).to_frame().set_index(y_null),
                y_notnull,
            ],
            axis=0,
        ).rename(columns={0: self.target})
        X[self.target] = pred
        return X

    def fit(self, X):
        if not self.group:
            self.group = list(X.columns)
        cols = self.group + [self.target]
        x = X[cols].fillna(value=np.nan)
        y = x[self.target]
        y_notnull = y[y.notnull()]
        imp = SimpleImputer(strategy=self.seed_strategy)
        self.imputer = imp.fit(x.loc[:, self.group])
        non_null_data = pd.DataFrame(
            imp.fit_transform(x.loc[:, self.group]), index=x.index.values
        )
        x_notnull = non_null_data.loc[y.notnull(), :]
        if y_notnull.nunique() > 2:
            model = LinearRegression()
            model.fit(x_notnull, y_notnull)
            print(model.coef_)
            self.model = model
        else:
            model = LogisticRegression()
            model.fit(x_notnull, y_notnull)
            self.model = model
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)
