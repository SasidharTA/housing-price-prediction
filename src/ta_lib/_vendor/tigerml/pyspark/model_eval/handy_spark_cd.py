import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect import signature
from operator import itemgetter
from pyspark.mllib.common import (  # common extention in handyspark
    JavaModelWrapper,
    _java2py,
    _py2java,
)
from pyspark.mllib.evaluation import (
    BinaryClassificationMetrics,
    MulticlassMetrics,
)
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql.types import DoubleType, StructField, StructType

mpl.rc("lines", markeredgewidth=0.5)


def call2(self, name, *a):
    """Call method for JavaModelWrapper.

    This method should be used whenever the JavaModel returns a Scala Tuple
    that needs to be deserialized before converted to Python.
    """
    serde = self._sc._jvm.org.apache.spark.mllib.api.python.SerDe
    args = [_py2java(self._sc, a) for a in a]
    java_res = getattr(self._java_model, name)(*args)
    java_res = serde.fromTuple2RDD(java_res)
    res = _java2py(self._sc, java_res)
    return res


JavaModelWrapper.call2 = call2


# instead of from handyspark.plot import roc_curve, pr_curve
def pr_curve(precision, recall, pr_auc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = (
        {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    )
    ax.step(
        recall,
        precision,
        color="b",
        alpha=0.2,
        where="post",
        label="PR curve (area = %0.4f)" % pr_auc,
    )
    ax.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="lower left")
    ax.set_title("Precision-Recall Curve")
    return ax


def roc_curve(fpr, tpr, roc_auc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.4f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic Curve")
    ax.legend(loc="lower right")
    return ax


def thresholds(self):
    """Thresholds in descending order."""
    thr = np.linspace(1, 0, 101)
    return self._sc.parallelize(thr)


def roc(self):
    """Call the `roc` method from the Java class.

    * Returns the receiver operating characteristic (ROC) curve,
    * which is an RDD of (false positive rate, true positive rate)
    * with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
    * @see <a href="http://en.wikipedia.org/wiki/Receiver_operating_characteristic">
    * Receiver operating characteristic (Wikipedia)</a>
    """
    return self.call2("roc")


def pr(self):
    """Call the `pr` method from the Java class.

    * Returns the precision-recall curve, which is an RDD of (recall, precision),
    * NOT (precision, recall), with (0.0, p) prepended to it, where p is the precision
    * associated with the lowest recall on the curve.
    * @see <a href="http://en.wikipedia.org/wiki/Precision_and_recall">
    * Precision and recall (Wikipedia)</a>
    """
    return self.call2("pr")


def fMeasureByThreshold(self, beta=1.0):
    """Call the `fMeasureByThreshold` method from the Java class.

    * Returns the (threshold, F-Measure) curve.
    * @param beta the beta factor in F-Measure computation.
    * @return an RDD of (threshold, F-Measure) pairs.
    * @see <a href="http://en.wikipedia.org/wiki/F1_score">F1 score (Wikipedia)</a>
    """
    return self.call2("fMeasureByThreshold", beta)


def precisionByThreshold(self):
    """Call the `precisionByThreshold` method from the Java class.

    * Returns the (threshold, precision) curve.
    """
    return self.call2("precisionByThreshold")


def recallByThreshold(self):
    """Call the `recallByThreshold` method from the Java class.

    * Returns the (threshold, recall) curve.
    """
    return self.call2("recallByThreshold")


def getMetricsByThreshold(self):
    """Obtain metrics by threshold (FPR, Recall and Precision).

    Returns
    -------
    metrics: DataFrame
    """
    thresholds = self.call("thresholds").collect()
    roc = self.call2("roc").collect()[0:-1]
    pr = self.call2("pr").collect()[0:]
    metrics = list(
        zip(
            thresholds,
            map(itemgetter(0), roc),
            map(itemgetter(1), roc),
            map(itemgetter(1), pr),
        )
    )
    metrics += [(0.0, 1.0, 1.0, 0.0)]
    sql_ctx = SQLContext.getOrCreate(self._sc)
    df = sql_ctx.createDataFrame(metrics).toDF(
        "threshold", "fpr", "recall", "precision"
    )
    return df


def confusionMatrix(self, threshold=0.5):
    """Generate the confusion matrix: predicted classes are in columns, they are ordered by class label ascending, as in "labels".

    Predicted classes are computed according to informed threshold.

    Parameters
    ----------
    threshold: double, optional
        Threshold probability for the positive class.
        Default is 0.5.

    Returns
    -------
    confusionMatrix: DenseMatrix
    """
    scoreAndLabels = self.call2("scoreAndLabels").map(
        lambda t: (float(t[0] > threshold), t[1])
    )
    mcm = MulticlassMetrics(scoreAndLabels)
    return mcm.confusionMatrix()


def print_confusion_matrix(self, threshold=0.5):
    """Print the confusion matrix.

    Predicted classes are in columns, they are ordered by class label ascending, as in "labels".
    Predicted classes are computed according to informed threshold.

    Parameters
    ----------
    threshold: double, optional
        Threshold probability for the positive class.
        Default is 0.5.

    Returns
    -------
    confusionMatrix: pd.DataFrame
    """
    cm = self.confusionMatrix(threshold).toArray()
    df = pd.concat([pd.DataFrame(cm)], keys=["Actual"], names=[])
    df.columns = pd.MultiIndex.from_product([["Predicted"], df.columns])
    return df


def plot_roc_curve(self, ax=None):
    """Plot of Receiver Operating Characteristic (ROC) curve.

    Parameter
    ---------
    ax: matplotlib axes object, default None
    """
    metrics = self.getMetricsByThreshold().toPandas()
    return roc_curve(metrics.fpr, metrics.recall, self.areaUnderROC, ax)


def plot_pr_curve(self, ax=None):
    """Plot of Precision-Recall (PR) curve.

    Parameter
    ---------
    ax: matplotlib axes object, default None
    """
    metrics = self.getMetricsByThreshold().toPandas()
    return pr_curve(metrics.precision, metrics.recall, self.areaUnderPR, ax)


def __init__(self, scoreAndLabels, scoreCol="score", labelCol="label"):
    if isinstance(scoreAndLabels, DataFrame):
        scoreAndLabels = scoreAndLabels.select(scoreCol, labelCol).rdd.map(
            lambda row: (float(row[scoreCol][1]), float(row[labelCol]))
        )

    sc = scoreAndLabels.ctx
    sql_ctx = SQLContext.getOrCreate(sc)
    df = sql_ctx.createDataFrame(
        scoreAndLabels,
        schema=StructType(
            [
                StructField("score", DoubleType(), nullable=False),
                StructField("label", DoubleType(), nullable=False),
            ]
        ),
    )

    java_class = sc._jvm.org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    java_model = java_class(df._jdf)
    super(BinaryClassificationMetrics, self).__init__(java_model)


BinaryClassificationMetrics.__init__ = __init__
BinaryClassificationMetrics.thresholds = thresholds
BinaryClassificationMetrics.roc = roc
BinaryClassificationMetrics.pr = pr
BinaryClassificationMetrics.fMeasureByThreshold = fMeasureByThreshold
BinaryClassificationMetrics.precisionByThreshold = precisionByThreshold
BinaryClassificationMetrics.recallByThreshold = recallByThreshold
BinaryClassificationMetrics.getMetricsByThreshold = getMetricsByThreshold
BinaryClassificationMetrics.confusionMatrix = confusionMatrix
BinaryClassificationMetrics.plot_roc_curve = plot_roc_curve
BinaryClassificationMetrics.plot_pr_curve = plot_pr_curve
BinaryClassificationMetrics.print_confusion_matrix = print_confusion_matrix
