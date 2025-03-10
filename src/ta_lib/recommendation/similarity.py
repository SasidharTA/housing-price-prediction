import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SimilarityRecommender:
    """Provide a similarity based approach for recommendation using user X item matrix including latent features as well.

    This is a memory based approach which uses similarity based on the user X item interaction along with the latent features of items as well

    Parameters
    ----------
    df : dataframe
        entire dataset
    all_items_set : dataframe
        all items set to be used for creating columns for user-item

    Examples
    --------
    >>> train = dataset.load_dataset(context, "train/trained_data")
    >>> test = dataset.load_dataset(context, "test/test_data")
    >>> train_util, test_util = evaluate.generate_train_test_util_matrix(
        [train, test], [<userid_col>], [<itemid_col>], <date_col>
        )
    >>> from ta_lib.recommendation.similarity import SimilarityRecommender
    >>> knn_recommender = SimilarityRecommender(train_util, train.<item_column>)
    >>> knn_recommender.fit()
    >>> recommendations = rec2.recommend(test_util.index[:5].values, 100)
    """

    def __init__(
        self,
        df,
        all_item_set,
    ):
        self.model = None
        self.df = df
        self.user_col = df.index.name
        # self.n_recommendations = n_recommendations
        # self.df[self.user_col] = self.df[self.user_col].astype(str)
        self.products_set = list(map(str, list(set(all_item_set))))

    def fit(
        self,
        n_neighbors=10,
        metric="cosine",
        algorithm="brute",
    ):
        """
        Train the model.

        Parameters
        ----------
        users: list
               list of users to recommend products
        n_recommendations: int
                           number of products to recommend
        n_neighbors: int
                     number of neighbors to consider for calculation,
        metric: string
                distance metric to be considered,
        algorithm: string
                   Algorithm used to compute the nearest neighbors:
                    'ball_tree' will use BallTree
                    'kd_tree' will use KDTree
                    'brute' will use a brute-force search.
                    'auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method.
        """
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1,
        )
        self.model = self.model.fit(self.df.values)
        distances, self.indices = self.model.kneighbors(self.df)
        self.indices = self.__convert_to_df(self.indices)

    def __convert_to_df(self, indices):
        """Convert model results; vector array to dataframe.

        Parameters
        ----------
        indices : np.array
                  numpy array of number_of_users X n_neighbors

        Returns
        -------
        pandas.DataFrame
            numpy array converted to dataframe
        """
        for i in range(len(indices)):
            indices[i] = self.df.iloc[indices[i], :].index.values
        indices = np.pad(indices, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        for i in range(len(indices)):
            indices[i][0] = self.df.index.values[i]
        indices = pd.DataFrame(
            indices, columns=[self.user_col] + list(range(self.model.n_neighbors))
        )
        indices = indices.astype(str)
        return indices

    def __add_propensity_scores(self, sub_df, user_ids, similar_ids):
        """
        __add_propensity_scores function adds propensity scores based on similar users for recommendations.

        Parameters
        ----------
        sub_df: pandas.DataFrame
                Utility matrix
        userId: string
                User Id for which propensity scores needs to be calculated
        similarIds: list
                    top n similar user

        Returns
        -------
        pandas.DataFrame
            propensity scores
        """

        x1 = sub_df[sub_df[self.user_col].isin(similar_ids)]
        a = x1.set_index(self.user_col).mean()
        a = x1.set_index(self.user_col).mean()
        a = a[a > 0]

        x1[a.index] = x1[a.index].replace({0: np.nan}).fillna(a.to_dict())
        return x1[x1[self.user_col] == user_ids]

    def recommend(self, users, n_recommendations=10):
        """Generate top n recommended products based on highest propensity score for each user.

        Parameters
        ----------
        users: list
               list of users to recommend
        n_recommendations: int
                           Number of products to recommend

        Returns
        -------
        pandas.DataFrame
            recommendations
        """

        if not self.model:
            return "Error - Model not trained."

        if n_recommendations <= 0:
            return "Error - Value of n_recommendations < 0"

        # using np array for faster computation
        recommendation_array = None
        df2 = self.df.loc[:, self.products_set].reset_index()
        for i in users:
            i = str(i)
            z = self.__add_propensity_scores(
                df2,
                i,
                self.indices.loc[
                    self.indices[self.user_col] == i, self.indices.columns[1:]
                ].values[0],
            )
            if recommendation_array is None:
                recommendation_array = z
            else:
                recommendation_array = np.vstack([recommendation_array, z])
        recommendation_array = pd.DataFrame(recommendation_array, columns=df2.columns)

        recommendation_array = recommendation_array.replace(1, -1).set_index(
            self.user_col
        )

        # recommendation_results = recommendation_array.agg(
        #     lambda s: pd.Series(
        #         [*s.nlargest(k).index, *s.nlargest(k)],
        #         (
        #             [f"product{i+1}" for i in range(k)]
        #             + [f"product{i+1}_propensity" for i in range(k)]
        #         ),
        #     ),
        #     axis="columns",
        # )

        recommendation_results = (
            recommendation_array.agg(
                lambda s: dict(
                    zip(
                        [*s.nlargest(n_recommendations).index],
                        [*s.nlargest(n_recommendations)],
                    )
                ),
                axis="columns",
            )
            .to_frame()
            .rename(columns={0: "recommendations"})
            .reset_index()
        )

        return recommendation_results
