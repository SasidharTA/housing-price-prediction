from sklearn.pipeline import Pipeline, make_pipeline

from .helpers import *


class BackendMixin:
    """Backend Mixin class."""

    @property
    def backend(self):
        """Returns data module for Backend Mixin class."""
        return self._data.__module__.split(".")[0]

    def __getattr__(self, item):
        if "_data" in dir(self):
            if self.backend == BACKENDS.dask:
                if item == "iloc":
                    return DaskiLocIndexer(self)
                elif item == "T":
                    return convert_to_tiger_assets(self._data.compute().T)
            if (
                self.backend == BACKENDS.dask
                and not hasattr(self._data, item)
                and hasattr(pd.DataFrame, item)
            ):
                attr = getattr(pd.DataFrame, item)
                if callable(attr):
                    attr = tigerify(attr)
                print("Using map_partitions to execute {}".format(item))
                return daskify_pandas(self, attr)
            else:
                attr = getattr(self._data, item)
                if (
                    callable(attr)
                    and getattr(self._data.__class__, item).__class__.__name__
                    != "property"
                ):
                    attr = tigerify(attr)
                else:
                    attr = convert_to_tiger_assets(attr)
                return attr

    def __setattr__(self, key, value):
        if hasattr(self._data, key):
            self._data.__setattr__(key, value)
        else:
            self.__dict__[key] = value

    def __setitem__(self, key, value):
        value = detigerify(value)
        self._data.__setitem__(key, value)

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, *args):
        args, kwargs = detigerify_inputs(args)
        try:
            result = self._data.__getitem__(*args)
        except Exception as e:
            raise Exception("getitem failed for args - {}. Error - {}".format(args, e))
        return convert_to_tiger_assets(result)

    def __len__(self):
        return self._data.__len__()

    def __str__(self):
        return self._data.__str__()

    def compute(self):
        """Computes for Backend Mixin class."""
        from tigerml.core.utils import compute_if_dask

        return convert_to_tiger_assets(compute_if_dask(self._data))

    def persist(self):
        """Persists for Backend Mixin class."""
        from tigerml.core.utils import persist_if_dask

        return convert_to_tiger_assets(persist_if_dask(self._data))

    @property
    def shape(self):
        """Returns shape of dataframe for Backend Mixin class."""
        if self.backend == BACKENDS.dask:
            import dask

            return dask.compute(self._data.shape)[0]
        return self._data.shape

    @property
    def empty(self):
        """Checks if dataframe is empty for Backend Mixin class."""
        if self.backend == BACKENDS.dask:
            return len(self._data.index) == 0 or len(self._data.columns) == 0
        return self._data.empty

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __ge__(self, other):
        return self.ge(other)

    def __gt__(self, other):
        return self.gt(other)

    def __le__(self, other):
        return self.le(other)

    def __lt__(self, other):
        return self.lt(other)

    def __and__(self, other):
        other = detigerify(other)
        result = self._data.__and__(other)
        return convert_to_tiger_assets(result)

    def __or__(self, other):
        other = detigerify(other)
        result = self._data.__or__(other)
        return convert_to_tiger_assets(result)

    def __add__(self, other):
        other = detigerify(other)
        return self._data.__add__(other)

    def __radd__(self, other):
        other = detigerify(other)
        return self._data.__radd__(other)

    def __sub__(self, other):
        other = detigerify(other)
        return self._data.__sub__(other)

    def __rsub__(self, other):
        other = detigerify(other)
        return self._data.__rsub__(other)

    def __mul__(self, other):
        other = detigerify(other)
        return convert_to_tiger_assets(self._data.__mul__(other))

    def __rmul__(self, other):
        other = detigerify(other)
        return self._data.__rmul__(other)

    def __div__(self, other):
        other = detigerify(other)
        return convert_to_tiger_assets(self._data.__div__(other))

    def __rdiv__(self, other):
        other = detigerify(other)
        return self._data.__rdiv__(other)

    def __floordiv__(self, other):
        other = detigerify(other)
        return self._data.__floordiv__(other)

    def __rfloordiv__(self, other):
        other = detigerify(other)
        return self._data.__rfloordiv__(other)

    def __truediv__(self, other):
        other = detigerify(other)
        return self._data.__truediv__(other)

    def __rtruediv__(self, other):
        other = detigerify(other)
        return self._data.__rtruediv__(other)

    def __divmod__(self, other):
        other = detigerify(other)
        return self._data.__divmod__(other)

    def __rdivmod__(self, other):
        other = detigerify(other)
        return self._data.__rdivmod__(other)

    def __neg__(self):
        return self._data.__neg__()

    def __bool__(self):
        return self._data.__bool__()


class DaskiLocIndexer:
    """Daski Loc Indexer."""

    def __init__(self, data):
        self._data = data._data

    def __getitem__(self, item):
        # import pdb
        # pdb.set_trace()
        # from collections.abc import Iterable
        if isinstance(item, tuple):
            rows, cols = item
        else:
            rows = item
        max_row = 0
        if isinstance(rows, slice):
            stop = rows.stop
            if rows.stop < 0:
                stop = len(self._data) + rows.stop
            max_row = max(max_row, stop)
        elif isinstance(rows, int):
            if rows < 0:
                rows = len(self._data) + rows
            max_row = max(max_row, rows)
        else:
            raise NotImplementedError
        result = self._data.head(max_row + 1).iloc[item]
        result = convert_to_tiger_assets(result)
        return result


class TAPipeline(Pipeline):
    """Ta pipeline class."""

    def __init__(self, pipeline=None, **kwargs):
        if pipeline:
            for att in [
                a
                for a in dir(pipeline)
                if not a.startswith("__") and a not in ["predict"]
            ]:
                try:
                    setattr(self, att, getattr(pipeline, att))
                except Exception:
                    pass
        else:
            self.__init__(Pipeline(**kwargs))
        self.score_ = 0

    @classmethod
    def from_parent(cls, pipeline):
        """Identifies from parent for Ta pipeline class."""
        steps = pipeline.steps
        new_obj = cls(steps=steps)
        for att in [
            a for a in dir(pipeline) if not a.startswith("__") and a not in ["predict"]
        ]:
            try:
                setattr(new_obj, att, getattr(pipeline, att))
            except Exception:
                pass
        return new_obj

    def predict(self, X, **predict_params):
        """Predicts using model."""
        # import pdb
        # pdb.set_trace()
        y = super().predict(X, **predict_params)
        return y.ravel()

    def get_step(self, step_index, only_object=True, only_name=False):
        """Gets step."""
        step = self.steps[step_index]
        if only_object:
            return step[1]
        if only_name:
            return step[0]
        else:
            return {"name": step[0], "object": step[1]}

    def get_steps_before_step(self, step_index):
        """Gets step before step."""
        preproc_steps = self.steps[:step_index]
        return preproc_steps

    def get_data_at_step(self, step_index, X):
        """Gets data at step."""
        preproc_steps = self.get_steps_before_step(step_index)
        if len(preproc_steps) > 0:
            for proc_name, proc in preproc_steps:
                new_x_train = pd.DataFrame(proc.transform(X))
                if not proc_name:  # don't change names if proc_name is empty
                    X = new_x_train
                    continue
                if len(new_x_train.columns) == len(X.columns):
                    X = new_x_train.rename(
                        columns=dict(
                            zip(
                                list(new_x_train.columns),
                                ["{}({})".format(proc_name, x) for x in X.columns],
                            )
                        )
                    )
                elif hasattr(proc, "get_feature_names"):
                    if proc.__class__.__name__ == "FeatureUnion":
                        X = new_x_train
                    else:
                        X = new_x_train.rename(
                            columns=dict(
                                zip(
                                    list(new_x_train.columns),
                                    [
                                        "{}({})".format(proc.__class__.__name__, x)
                                        for x in proc.get_feature_names(list(X.columns))
                                    ],
                                )
                            )
                        )
                elif hasattr(proc, "get_support"):
                    new_features = list(X.columns[proc.get_support()])
                    X = new_x_train.rename(
                        columns=dict(zip(new_x_train.columns, new_features))
                    )
                elif str(proc.__module__).startswith("tpot"):
                    if (
                        proc.__module__.startswith("tpot.builtins")
                        and proc.__class__.__name__ == "ZeroCount"
                    ):
                        X = new_x_train.rename(
                            columns=dict(
                                zip(
                                    list(new_x_train.columns),
                                    ["count_of_0", "count_of_non_0"] + list(X.columns),
                                )
                            )
                        )
                    elif (
                        proc.__module__.startswith("tpot.builtins")
                        and proc.__class__.__name__ == "StackingEstimator"
                    ):
                        estimator_class = proc.estimator.__class__.__name__
                        if hasattr(proc.estimator, "predict_proba"):
                            number_of_classes = (
                                len(list(new_x_train.columns))
                                - len(list(X.columns))
                                - 1
                            )
                            X = new_x_train.rename(
                                columns=dict(
                                    zip(
                                        list(new_x_train.columns),
                                        list(X.columns)
                                        + ["{}_prediction".format(estimator_class)]
                                        + [
                                            "{}_prob_for_{}".format(estimator_class, cl)
                                            for cl in range(0, number_of_classes)
                                        ],
                                    )
                                )
                            )
                        else:
                            X = new_x_train.rename(
                                columns=dict(
                                    zip(
                                        list(new_x_train.columns),
                                        list(X.columns)
                                        + ["{}_prediction".format(estimator_class)],
                                    )
                                )
                            )
                else:
                    X = new_x_train
                    print(
                        "WARNING: The shape of input features is changed. "
                        "Interpretability will be lost. Process is {}".format(
                            proc.__class__
                        )
                    )
        return X

    @property
    def feature_importances_(self):
        """Gets Feature importances."""
        estimator = self.get_step(-1)
        if hasattr(estimator, "estimator"):
            estimator = estimator.estimator
        if hasattr(estimator, "coef_"):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, "feature_importances_", None)
        if coefs is None:
            raise RuntimeError(
                "The estimator does not expose "
                '"coef_" or "feature_importances_" '
                "attributes"
            )
        return coefs

    @classmethod
    def from_string(cls, pipeline_string, x_train=None, y_train=None):
        """Makes the pipeline using pipeline string and fits model."""
        try:
            pipeline = make_pipeline(eval(pipeline_string))
        except Exception as e:
            raise Exception("Cannot create the pipeline. Error - {}".format(e))
        ta_pipeline = cls(pipeline)
        if x_train is not None and y_train is not None:
            ta_pipeline.fit(x_train, y_train)
        return ta_pipeline
