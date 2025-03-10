"""
ClusterMMDD() currently works for categorical data.

To proceed with numerical data, use a Discretizer to bin the data in numerical features before passing data into fit().
Ex. pd.qcut(data[col], q=[0, .25, .75, 1], labels=False).
# col represent numerical columns only.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime as dt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted


class MultiColumnLabelEncoder(TransformerMixin, BaseEstimator):
    """Multi columns label encoder class."""

    def __init__(self):
        """Multi columns label encoder class initializer."""
        pass

    def fit(self, X, y=None):
        """Encodes the x variable that is provided as parameter to function."""
        X = check_array(
            X,
            accept_sparse=False,
            dtype=object,
            copy=True,
            accept_large_sparse=False,
        )
        self._enc = {}
        min_ = 0
        for col in range(X.shape[1]):
            t = np.unique(X[:, col])
            self._enc[col] = dict(zip(t, range(min_, min_ + len(t))))
            min_ = min_ + len(t)
        return self

    def transform(self, X):
        """Transforms columns of X specified in self.columns using LabelEncoder().

        If no columns specified, transforms all
        columns in X.
        """
        X = check_array(
            X,
            accept_sparse=False,
            dtype=object,
            copy=True,
            accept_large_sparse=False,
        )
        self.fit(X)
        if X.shape[1] != len(self._enc):
            raise ValueError(
                "The number of features {} in transform is different from the number of features {} in fit.".format(
                    X.shape[1], len(self._enc)
                )
            )
        for col in self._enc:
            if len(set(X[:, col]) - set(self._enc[col])) > 0:
                raise ValueError(
                    "There are unseen levels in the transform data {}".format(
                        set(X[:, col]) - set(self._enc[col])
                    )
                )
            X[:, col] = np.vectorize(self._enc[col].get)(X[:, col])
        return X


class ClusterMMDD(ClusterMixin, TransformerMixin, BaseEstimator):
    """Perform Clustering by Mixture Models for Discrete Data.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form
    encode_labels : bool, default=True
        encoding discrete variables
    small_em_iter : int, default=20
    em_iter : int, default=15
    long_em_iter : int, default=5000
    tol : float, default=1e-8
    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.
    random_state : int, RandomState instance or None, default=0
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    log_lik_: float
        Log Likelihood
    pi_k_: ndarray of shape (n_cluster,)
        The vector of mixing proportions
    prob_: ndarray of shape (no of levels, n_cluster)
        A list of matrices, each matrix being the probabilities of a variable in different clusters.
    tik_: ndarray of shape (The size (number of lines) of the dataset, n_cluster)
        A stochastic matrix given the a posteriori membership probabilities.

    Examples
    --------
    >>> from tigerml.clustering.cmmdd import ClusterMMDD
    >>> import numpy as np
    >>> X = np.array([["A", "C"], ["A", "D"], ["A", "E"], ["B", "C"], ["B", "D"], ["B", "E"]])
    >>> cmmdd = ClusterMMDD(n_clusters=2, random_state=0).fit(X)
    2020-09-30 13:30:48.570462: Running small EM 1 out of 20 small EMs
    log likelihood: -10.750666241151901
    2020-09-30 13:30:48.581212: Running small EM 2 out of 20 small EMs
    log likelihood: -10.750556822453387
    2020-09-30 13:30:48.588441: Running small EM 3 out of 20 small EMs
    log likelihood: -10.75057200789978
    2020-09-30 13:30:48.606447: Running small EM 4 out of 20 small EMs
    log likelihood: -10.750592385592832
    2020-09-30 13:30:48.613835: Running small EM 5 out of 20 small EMs
    log likelihood: -10.750556815384533
    2020-09-30 13:30:48.620209: Running small EM 6 out of 20 small EMs
    log likelihood: -10.750591436748993
    2020-09-30 13:30:48.632043: Running small EM 7 out of 20 small EMs
    log likelihood: -10.752015751441448
    2020-09-30 13:30:48.636879: Running small EM 8 out of 20 small EMs
    log likelihood: -10.752677474309545
    2020-09-30 13:30:48.644317: Running small EM 9 out of 20 small EMs
    log likelihood: -10.75057999100893
    2020-09-30 13:30:48.651510: Running small EM 10 out of 20 small EMs
    log likelihood: -10.75055681536845
    2020-09-30 13:30:48.656429: Running small EM 11 out of 20 small EMs
    log likelihood: -10.750559126920452
    2020-09-30 13:30:48.669271: Running small EM 12 out of 20 small EMs
    log likelihood: -10.7506878513226
    2020-09-30 13:30:48.682628: Running small EM 13 out of 20 small EMs
    log likelihood: -10.750556815368329
    2020-09-30 13:30:48.689690: Running small EM 14 out of 20 small EMs
    log likelihood: -10.751619347639561
    2020-09-30 13:30:48.702618: Running small EM 15 out of 20 small EMs
    log likelihood: -10.750558486071702
    2020-09-30 13:30:48.709892: Running small EM 16 out of 20 small EMs
    log likelihood: -10.750556816938296
    2020-09-30 13:30:48.717224: Running small EM 17 out of 20 small EMs
    log likelihood: -10.753178465553406
    2020-09-30 13:30:48.722652: Running small EM 18 out of 20 small EMs
    log likelihood: -10.751263862367749
    2020-09-30 13:30:48.733206: Running small EM 19 out of 20 small EMs
    log likelihood: -10.75140699611315
    2020-09-30 13:30:48.740473: Running small EM 20 out of 20 small EMs
    log likelihood: -10.750682665411624
    2020-09-30 13:30:48.748284: Running a maximum of 5000 long run of EM ...
    log likelihood: -10.750556815368329
    >>> cmmdd.labels_
    array([1, 0, 0, 1, 0, 0])
    >>> cmmdd.predict([["A", "C"], ["B", "D"]])
    array([1, 0])

    References
    ----------
    Toussile W (2016). "ClustMMDD: Variable selection in Clustering by Mixture Models for Discret Data. R package version 1.0.4."
    """

    def __init__(
        self,
        n_clusters=8,
        small_em_iter=20,
        em_iter=15,
        long_em_iter=5000,
        tol=1e-8,
        random_state=None,
    ):
        """Cluster mdd class initializer."""
        self.n_clusters = n_clusters
        self.small_em_iter = small_em_iter
        self.em_iter = em_iter
        self.long_em_iter = long_em_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
        Fitted estimator.
        """
        X = self._check_X(X)
        self.data_ = X.copy()
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="unknown"),
            MultiColumnLabelEncoder(),
        )
        X = self.pipe_.fit_transform(X)
        self.X_ = X
        self.get_freq()
        self.shape_ = self.X_.shape
        self.pi_k_ = np.repeat(1.0 / self.n_clusters, self.n_clusters)
        self.tik_ = np.zeros([self.shape_[0], self.n_clusters])
        self.log_lik_ = None
        self.initialize_prob()
        self.small_em()
        print(
            "{d}: Running a maximum of {n} long run of EM ...".format(
                **{"d": dt.now(), "n": self.long_em_iter}
            )
        )
        d_lv = 10
        for _ in range(self.long_em_iter):
            lv_2 = self.log_lik_
            if d_lv > self.tol:
                self.expectation().maximisation()
                d_lv = (lv_2 - self.log_lik_) / self.log_lik_
                d_lv = -d_lv if (d_lv < 0) else d_lv
            else:
                print("log likelihood: {}".format(self.log_lik_))
                break
        self.labels_ = self.tik_.argmax(axis=1)
        self.is_fitted_ = True
        return self

    def _more_tags(self):
        """Returns dictionary of conditions to allow NaN for string and 2d array dtypes."""
        return {"allow_nan": True, "X_types": ["2darray", "string"]}

    def get_freq(self):
        """Gets values, counts, levels and frequency for each column in data."""
        self.values_ = []
        self.counts_ = []
        self.freqs_ = []
        self.levels_ = []
        for col in range(self.X_.shape[1]):
            values, counts = np.unique(self.X_[:, col], return_counts=True)
            freqs = counts / sum(counts)
            levels = int(len(values))
            self.values_ = np.append(self.values_, values)
            self.counts_ = np.append(self.counts_, counts)
            self.freqs_ = np.append(self.freqs_, freqs)
            self.levels_ = np.append(self.levels_, levels)
        self.levels_ = self.levels_.astype(int)
        self.counts_ = self.counts_.astype(int)
        return self

    def initialize_prob(self, random_state=None):
        """Assigns probability attribute to class object."""
        if random_state is not None:
            random_state = check_random_state(random_state)
        else:
            random_state = check_random_state(self.random_state)
        for i, v in enumerate(self.levels_):
            prob = random_state.uniform(high=1, low=0, size=[int(v), self.n_clusters])
            prob = prob / prob.sum(axis=0)
            if i == 0:
                self.prob_ = prob
            else:
                self.prob_ = np.vstack((self.prob_, prob))
        return self

    def expectation(self):
        """Calculates expectation."""
        for i in range(self.X_.shape[1]):
            if i == 0:
                self.tik_ = self.prob_[list(self.X_[:, i]), :]
            else:
                self.tik_ = self.tik_ * self.prob_[list(self.X_[:, i]), :]
        self.tik_ = self.tik_ * self.pi_k_
        self.log_lik_ = np.sum(np.log(self.tik_.sum(axis=1)))
        self.tik_ = self.tik_ / self.tik_.sum(axis=1, keepdims=True)
        return self

    def maximisation(self):
        """Gets maximuum probabilities for each level in columns."""
        s1 = self.tik_.sum(axis=0)
        self.pi_k_ = s1 / self.X_.shape[0]
        self.pi_k_[-1] = 1 - self.pi_k_[:-1].sum()
        s1 = self.pi_k_ * self.X_.shape[0]
        for j in range(self.X_.shape[1]):
            for lvl in np.unique(self.X_[:, j]):
                ind = self.X_[:, j] == lvl
                self.prob_[lvl, :] = self.tik_[ind, :].sum(axis=0) / s1
        self.reabse_probs()

        # Laplacian smoothing
        # for idx in range(self.prob_.shape[1]):
        #     self.prob_[self.prob_[:, idx] < (1 / s1[idx]), idx] = 1 / s1[idx]
        # self.reabse_probs()
        return self

    def small_em(self):
        """Runs small EM."""
        lv_ = self.get_log_lik()
        pi_k = self.pi_k_.copy()
        prob_ = self.prob_.copy()
        tik = self.tik_.copy()
        for i in range(self.small_em_iter):
            print(
                "{d}: Running small EM {ith} out of {n} small EMs".format(
                    **{"d": dt.now(), "ith": (i + 1), "n": self.small_em_iter}
                )
            )
            self.initialize_prob(
                random_state=self.random_state
            )  # to get different probabilities
            for _ in range(self.em_iter):
                self.expectation()
                self.maximisation()
            lv0 = self.get_log_lik()
            print("log likelihood: {}".format(lv0))
            if lv0 > lv_:
                lv_ = lv0
                pi_k = self.pi_k_.copy()
                prob_ = self.prob_.copy()
                tik = self.tik_.copy()
        self.pi_k_ = pi_k.copy()
        self.prob_ = prob_.copy()
        self.tik_ = tik.copy()
        self.log_lik_ = lv_
        return self

    def _check_X(self, X):
        X = check_array(
            X,
            accept_sparse=False,
            dtype=object,
            copy=True,
            accept_large_sparse=False,
            force_all_finite="allow-nan",
        )
        if isinstance(X, (pd.DataFrame)):
            X = X.values
        elif isinstance(X, (np.ndarray)):
            X = X
        elif isinstance(X, (list, tuple)):
            X = np.array(X)
        elif hasattr(X, ("__array__")):
            X = X.__array__()
        for i in range(X.shape[1]):
            try:
                X[:, i] = X[:, i].astype(str)
            except ValueError as err:
                raise ValueError(
                    "Data type not supported feature {} ".format(i) + ": {}".format(err)
                )
        return X

    def transform(self, X):
        """
        Checks if its fitted and ensures feature count is same. Transforms fitted data based on pipeline.

        Returns class labels
        """
        check_is_fitted(self, "is_fitted_")
        X = self._check_X(X)
        if X.shape[1] != self.data_.shape[1]:
            raise ValueError(
                "The number of features {}".format(X.shape[1])
                + "in transform is different from the number of features {} in fit.".format(
                    self.data_.shape[1]
                )
            )
        X = self.pipe_.transform(X)
        for i in range(X.shape[1]):
            if i == 0:
                tik = self.prob_[list(X[:, i]), :]
            else:
                tik = tik * self.prob_[list(X[:, i]), :]
        tik = tik * self.pi_k_
        tik = tik / tik.sum(axis=1, keepdims=True)
        class_labels = tik.argmax(axis=1)
        return class_labels

    def predict(self, X):
        """Predicts y using x by calling transform function."""
        return self.transform(X)

    def get_log_lik(self):
        """Returns log_lik_."""
        if self.log_lik_ is None:
            self.expectation()
        return self.log_lik_

    def reabse_probs(self):
        """Gets max cummulative probabilities."""
        cumlevels = np.cumsum(self.levels_)
        cumlevels = cumlevels.astype(int)
        for idx, lvl in enumerate(cumlevels):
            if idx == 0:
                self.prob_[(lvl - 1), :] = np.maximum(
                    [0] * self.n_clusters, 1 - self.prob_[0 : (lvl - 1), :].sum(axis=0)
                )
            else:
                self.prob_[(lvl - 1), :] = np.maximum(
                    [0] * self.n_clusters,
                    1
                    - self.prob_[int(cumlevels[(idx - 1)]) : (lvl - 1), :].sum(axis=0),
                )
        return self
