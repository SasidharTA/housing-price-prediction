import numpy as np
import pandas as pd
from surprise import (
    NMF,
    SVD,
    BaselineOnly,
    CoClustering,
    Dataset,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    NormalPredictor,
    Reader,
    SlopeOne,
    accuracy,
)
from surprise.model_selection import cross_validate


class ExplicitRecommender:
    """Gives ease of access to using surprise library and multiple models under it.

    Surprise is a library for easy to implement recommendations model. This class provides functions to access those functionalities with ease.

    Parameters
    ----------
    train : dataframe
        train data set
    test : dataframe
        test data set
    user_col : string
        specify the column which corresponds to user
    item_col : string
        specify the column which corresponds to item
    explicit_feedback_col : string
        explicit feedback, whether rating or something which gives sense of rating.
    rating_scale : tuple
        min, max value of ratings
    random_state : integer
        Set the random state variable

    Examples
    --------
    >>> train = dataset.load_dataset(context, "train/retail_trained")
    >>> test = dataset.load_dataset(context, "test/retail_test")
    >>> from ta_lib.recommendation.explicit import ExplicitRecommender
    >>> rec = ExplicitRecommender(train, test, <userid_col>, <itemid_col>, <explicit_feedback_coll>)
    >>> ## check this for description
    >>> ## https://benfred.github.io/implicit/api/models/cpu/als.html
    >>> params = {
            "n_factors": 100,
            "n_epochs": 20,
            "biased": True,
            "init_mean": 0,
            "init_std_dev": 0.1,
            "lr_all": 0.005,
            "reg_all": 0.02,
            "lr_bu": None,
            "lr_bi": None,
            "lr_pu": None,
            "lr_qi": None,
            "reg_bu": None,
            "reg_bi": None,
            "reg_pu": None,
            "reg_qi": None,
            "random_state": 7,
            "verbose": False,
        }
    >>> exp_rec.fit("svd", params=params)
    >>> recommendations = exp_rec.recommend(
            <userid_list>, <itemid_list>, n_recommendations=100
        )
    """

    def __init__(
        self,
        train,
        test,
        user_col,
        item_col,
        explicit_feedback_col,
        rating_scale=(1, 5),
        random_state=7,
    ):
        self.random_state = random_state
        reader = Reader(rating_scale=rating_scale)
        self.products = list(set(train[item_col]))
        self.train = Dataset.load_from_df(
            train[[user_col, item_col, explicit_feedback_col]], reader
        )
        self.test = Dataset.load_from_df(
            test[[user_col, item_col, explicit_feedback_col]], reader
        )

        self.train_ratings = self.train.build_full_trainset()
        self.test_ratings = self.test.build_full_trainset().build_testset()

    def compare_models(self):
        """Compare all the models results.

        Returns
        -------
        pandas.DataFrame
            with model comparison results

        """
        benchmark = []
        # Iterate over all algorithms
        algorithms = [
            SVD(),
            SlopeOne(),
            NMF(),
            NormalPredictor(),
            KNNBaseline(),
            KNNBasic(),
            KNNWithMeans(),
            KNNWithZScore(),
            BaselineOnly(),
            CoClustering(),
        ]

        print("Attempting: ", str(algorithms), "\n\n\n")

        for algorithm in algorithms:
            print("Starting: ", str(algorithm))
            # Perform cross validation
            results = cross_validate(
                algorithm,
                self.train,
                measures=["rmse", "mae"],
                cv=3,
                verbose=False,
                n_jobs=-1,
            )
            # results = cross_validate(algorithm, data, measures=['RMSE','MAE'], cv=3, verbose=False)

            # Get results & append algorithm name
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = pd.concat(
                [
                    tmp,
                    pd.Series(
                        [str(algorithm).split(" ")[0].split(".")[-1]],
                        index=["Algorithm"],
                    ),
                ]
            )
            benchmark.append(tmp)
            print("Done: ", str(algorithm), "\n\n")
        surprise_results = (
            pd.DataFrame(benchmark).set_index("Algorithm").sort_values("test_rmse")
        )
        return surprise_results

    def fit(self, model_name, params={}):
        """Train the model with given set of params.

        Parameters
        ----------
        model_name: string
                    name of the model to be used in lowercase
        params: dict
                set of parameters to be passed to the respective model

        Examples
        --------
        >>> models = {
        >>>    "svd": SVD,
        >>>    "svdpp": SVDpp,
        >>>    "slopeone": SlopeOne,
        >>>    "nmf": NMF,
        >>>    "normalpredictor": NormalPredictor,
        >>>    "knnbaseline": KNNBaseline,
        >>>    "knnbasic": KNNBasic,
        >>>    "knnwithmeans": KNNWithMeans,
        >>>    "knnwithzscore": KNNWithZScore,
        >>>    "baselineonly": BaselineOnly,
        >>>    "coclustering": CoClustering,
        >>>    }

        """

        models = {
            "svd": SVD,
            "slopeone": SlopeOne,
            "nmf": NMF,
            "normalpredictor": NormalPredictor,
            "knnbaseline": KNNBaseline,
            "knnbasic": KNNBasic,
            "knnwithmeans": KNNWithMeans,
            "knnwithzscore": KNNWithZScore,
            "baselineonly": BaselineOnly,
            "coclustering": CoClustering,
        }
        if "random_state" not in params:
            params["random_state"] = 7
        model_name = model_name.lower()
        self.model = models[model_name](**params)
        self.model.fit(self.train_ratings)

    def recommend(self, users, items, n_recommendations=10):
        """Predicts the product for given users.

        Parameters
        ----------
        users: list
               list of users to recommend products
        n_recommendations: int
                           number of products to recommend

        Returns
        -------
        pandas.DataFrame
            recommendations in Dataframe
        """
        results = []
        for user in users:
            recommendations = np.array(
                sorted(
                    [self.model.predict(uid=user, iid=i, verbose=False) for i in items],
                    key=lambda x: x.est,
                    reverse=True,
                )
            )[:, [1, 3]]
            results.append(
                dict(
                    recommendations
                    if n_recommendations <= 0
                    else recommendations[:n_recommendations]
                )
            )
        return pd.DataFrame({"user": users, "recommendations": results})

    def get_scores(self):
        """Print model evaluation results on test data split based on RMSE and MAE."""
        train_predictions = self.model.test(self.train_ratings.build_testset())
        test_predictions = self.model.test(self.test_ratings)
        print(
            "RMSE on training data : ",
            accuracy.rmse(train_predictions, verbose=False),
        )
        print(
            "MAE on training data : ",
            accuracy.mae(train_predictions, verbose=False),
        )
        print("RMSE on test data: ", accuracy.rmse(test_predictions, verbose=False))
        print("MAE on test data: ", accuracy.mae(test_predictions, verbose=False))
