import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from . import evaluate


class ImplicitRecommender:
    """
    Gives ease of access to using Implicit library, currently only ALS is supported.

    Implicit is a library for easy to implement recommendations model with implicit feedback.

    Parameters
    ----------
    train: dataframe
        train data set
    test: dataframe
        test data set
    user_col: string
        specify the column which corresponds to user
    item_col: string
        specify the column which corresponds to item
    implicit_feedback_col : string
        implicit feedback, something which gives sense of rating or either 0/1.
    random_state: integer
        Set the random state variable

    Examples
    --------
    >>> train = dataset.load_dataset(context, "train/retail_trained")
    >>> test = dataset.load_dataset(context, "test/retail_test")
    >>> from ta_lib.recommendation.implicit import ImplicitRecommender
    >>> rec = ImplicitRecommender(train, test, <userid_col>, <itemid_col>, <implicit_feedback_coll>)
    >>> ## check this for description
    >>> ## https://benfred.github.io/implicit/api/models/cpu/als.html
    >>> params = {
            "factors": 100,
            "regularization": 0.01,
            "alpha": 1.0,
            "use_native": True,
            "use_cg": True,
            "iterations": 100,
            "calculate_training_loss": False,
            "num_threads": 0,
            "random_state": 20,
        }
    >>> rec.fit(params=params)
    >>> result = rec.recommend(<userid_list>, 100)
    """

    def __init__(
        self,
        train,
        test,
        user_col,
        item_col,
        implicit_feedback_col,
        random_state=7,
    ):
        self.__test_data = test.copy()
        self.__train_data = train.copy()
        self.__user_col = user_col
        self.__item_col = item_col
        unique_customers = train[user_col].append(test[user_col]).unique()
        self.customer_ids = dict(
            zip(
                unique_customers,
                np.arange(unique_customers.shape[0], dtype=np.int32),
            )
        )

        unique_items = train[item_col].append(test[item_col]).unique()
        self.item_ids = dict(
            zip(unique_items, np.arange(unique_items.shape[0], dtype=np.int32))
        )

        # Evaluate.__init__(self)
        self.random_state = random_state
        train = self.__get_continous_ids(train, user_col, item_col)
        self.train = self.__to_user_item_csr(train, implicit_feedback_col)

        test = self.__get_continous_ids(test, user_col, item_col)
        self.test = self.__to_user_item_csr(test, implicit_feedback_col)
        print("done : preparing data")

    def __get_continous_ids(self, df, user_col, item_col):
        df["customer_id"] = df[user_col].apply(lambda i: self.customer_ids[i])
        df["item_id"] = df[item_col].apply(lambda i: self.item_ids[i])

        return df

    def __to_user_item_csr(self, df, implicit_rating_col):
        """Turn a dataframe with transactions into a CSR sparse users X items matrix.

        Parameters
        ----------
        df: pandas.DataFrame
            dataset with user product and implicit feedback details.

        implicit_rating_col: str
            name of implicit feedback column.

        Returns
        -------
        sparse.csr matrix
        """
        row = df["customer_id"].values
        col = df["item_id"].values
        # data = np.ones(df.shape[0])
        data = df[implicit_rating_col].astype(float).values
        csr = sparse.csr_matrix((data, (row, col)))
        return csr

    def fit(self, params={}):
        """Train model using ALS algorithm.

        Parameters
        ----------
        params: dict
            set of parameters to be passed to model

        Returns
        -------
        model
            trained model
        """
        # params hard coded for testing, change it to **params while finalizing
        if "random_state" not in params:
            params["random_state"] = 7
        self.model = implicit.als.AlternatingLeastSquares(**params)
        self.model.fit(self.train)
        return self.model

    def recommend(self, users, n_recommendations=100):
        """Predict the recommendation for users.

        Parameters
        ----------
        users: list
            list of users to recommend products
        n_recommendations: int
            number of products to recommend

        Returns
        -------
        pandas.DataFrame
            Recommendation Dataframe

        """
        item_ids_mappings = {v: k for k, v in self.item_ids.items()}
        # customer_ids_mappings = {v: k for k, v in self.customer_ids.items()}
        users_mapped = [self.customer_ids[i] for i in users]
        r_ids, r_scores = self.model.recommend(
            users_mapped, self.train[users_mapped], N=n_recommendations
        )
        recommendations = []
        for i, val in enumerate(users_mapped):
            recommendations.append(
                dict(zip([item_ids_mappings[j] for j in r_ids[i]], r_scores[i]))
            )
        df_result = pd.DataFrame({"user": users, "recommendations": recommendations})
        return df_result

    def get_scores(self, n_recommendations=10):
        """Print model evaluation results on test data split based on MAPE.

        Parameters
        ----------
        n_recommendations: int
            number of products to recommend
        """
        result = self.recommend(
            self.__test_data[self.__user_col].unique(), n_recommendations
        )
        # group all products of user in test data
        test_data = (
            self.__test_data.groupby(self.__user_col)
            .agg({self.__item_col: lambda x: list(set(x))})
            .reset_index()
        )

        # create a dataframe of recommended products and existing proeducts
        df_evaluate = pd.merge(
            test_data, result, left_on=self.__user_col, right_on="user", how="inner"
        )
        # Create variables for storing relevant_products and predicted_variables
        relevant_products = df_evaluate[self.__item_col].tolist()
        predicted_products = df_evaluate["recommendations"].tolist()

        # map@k, k can be set here as well
        score = evaluate.mean_average_precision_at_k(
            relevant_products,
            [list(i.keys()) for i in predicted_products],
            k=n_recommendations,
        )
        print(f"MAP@{n_recommendations} : ", score)
