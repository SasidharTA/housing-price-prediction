import itertools
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn.functional as functional
import torch_geometric.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear, SAGEConv, to_hetero


class GNNEncoder(torch.nn.Module):
    """GNN Encoder class for embeddings generation."""

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        """Feed forward."""

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    """Edge Decoder class for embeddings decoding."""

    def __init__(self, hidden_channels, node_types):
        super().__init__()
        self.user_node_name = node_types[0]
        self.item_node_name = node_types[1]
        # print("decoder", node_types)
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        """Feed forward."""

        row, col = edge_label_index
        z = torch.cat(
            [z_dict[self.user_node_name][row], z_dict[self.item_node_name][col]], dim=-1
        )
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    """Graph sage model class."""

    def __init__(self, graph, hidden_channels):
        super().__init__()
        self.graph = graph
        node_types, edge_types = graph.metadata()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, graph.metadata(), aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels, node_types)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        """Feed forward."""

        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


class Graphsage:
    """
    Generate recommendations using graphsage and GNN based approach.

    The basic intuition is the following: the available data might be better represented in a graph. GNNs can leverage both content information (user & item features) as well as graph structure (user-item interaction), whereas typically, traditional models can leverage only one of the two.

    Parameters
    ----------
    user_mapping : pandas.dataframe
                Dataset with first column as mapped userid and second column a original userid
                --note : column name should exactly in this format ['mapped_<col_name>','<col_name>']

    item_mapping : pandas.dataframe
        Dataset with first column as mapped itemid and second column a original itemid
        --note : column name should exactly in this format ['mapped_<col_name>','<col_name>']

    user_nodes_features : pandas.dataframe or numpy_array
        Dataset of user features sorted based on mapped user ids

    item_nodes_features : pandas.dataframe or numpy_array
        Dataset of item features sorted based on mapped user ids

    edge_data : pandas.dataframe
        Dataset with edge details and ratings
        --note : edges needs to be created using mapped ids for example
                    [mapped_userid,mapped_itemid,ratings]
                    where userid and itemid were original column names

    feedback_col : string
        Name of the feedback column

    ratings : list [int,int]
        List showing min and max rating available in the dataset

    seed : int, optional
        Seed value to set for replicating same results

    validation_frac : float, optional
        Fraction value which will be considered as validation set

    test_fraction : float, optional
        Fraction value which will be considered as test set

    neg_sampling_ratio : float
        Fraction of negative sample to create for training
        negative samples are non existent edges in the graph

    Examples
    --------
    >>> from ta_lib.recommendation.graphsage import Graphsage
    >>> graph = Graphsage(
            user_map,
            item_map,
            user_features,
            item_features,
            edge_data,
            feedback_col,
            feedback_range,
            scale_rating=False,
        )
    >>> results, model = graph.fit(n_epochs=200, e_patience=100, min_acc=0.001)
    >>> recommendations = graph.recommend(<userid_list>, n_recommendations=10)
    """

    def __init__(
        self,
        user_mapping,
        item_mapping,
        user_nodes_features,
        item_nodes_features,
        edge_data,
        feedback_col,
        ratings,
        scale_rating=False,
        seed=7,
        validation_frac=0.1,
        test_fraction=0.1,
        neg_sampling_ratio=0.0,
    ):
        """
        Create graph and split it into train, test, val dataset.

        Parameter
        ---------
        user_mapping : pandas.dataframe
            Dataset with first column as mapped userid and second column a original userid
            --note : column name should exactly in this format ['mapped_<col_name>','<col_name>']
        item_mapping : pandas.dataframe
            Dataset with first column as mapped itemid and second column a original itemid
            --note : column name should exactly in this format ['mapped_<col_name>','<col_name>']
        user_nodes_features : pandas.dataframe or numpy_array
            Dataset of user features sorted based on mapped user ids
        item_nodes_features : pandas.dataframe or numpy_array
            Dataset of item features sorted based on mapped user ids
        edge_data : pandas.dataframe
            Dataset with edge details and ratings
            --note : edges needs to be created using mapped ids for example
                        [mapped_userid,mapped_itemid,ratings]
                        where userid and itemid were original column names
        feedback_col : string
            Name of the feedback column
        ratings : list [int,int]
            List showing min and max rating available in the dataset
        seed : int, optional
            Seed value to set for replicating same results
        validation_frac : float, optional
            Fraction value which will be considered as validation set
        test_fraction : float, optional
            Fraction value which will be considered as test set
        neg_sampling_ratio : float
            Fraction of negative sample to create for training
            negative samples are non existent edges in the graph
        """
        self.seed = seed
        self.ratings = ratings
        random.seed(seed)
        np.random.seed(seed)
        self.__user_mapping = user_mapping
        self.__item_mapping = item_mapping
        self.__user_nodes_features = user_nodes_features
        self.__item_nodes_features = item_nodes_features
        self.scale_rating = scale_rating

        if self.scale_rating:
            self.__scaler = MinMaxScaler(feature_range=(min(ratings), max(ratings)))
            edge_data[feedback_col] = self.__scaler.fit_transform(
                edge_data[feedback_col].values.reshape(-1, 1)
            ).flatten()

        self.__edge_data = edge_data
        self.__feedback_col = feedback_col
        self.graph = self.create_graph(
            self.__user_nodes_features,
            self.__item_nodes_features,
            self.__edge_data,
            self.__user_mapping.columns[0],
            self.__item_mapping.columns[0],
            self.__feedback_col,
        )
        self.__node_types, self.edge_types = self.graph.metadata()
        self.__user_node_name = self.__node_types[0]
        self.__item_node_name = self.__node_types[1]
        self.train_val_test_split(validation_frac, test_fraction, neg_sampling_ratio)

    def create_graph(
        self,
        user_features,
        item_features,
        dataset,
        user_col_name,
        item_col_name,
        feedback_col_name,
    ):
        """
        Prepare graph structure that model learns from.

        Parameters
        ----------
        user_features : pandas.dataframe or numpy_array
            Dataset of user features sorted based on mapped user ids
        item_features : pandas.dataframe or numpy_array
            Dataset of item features sorted based on mapped user ids
        dataset : pandas.dataframe
            Dataset with edge details and ratings
            --note : edges needs to be created using mapped ids for example
                     [mapped_userid,mapped_itemid,ratings]
                     where userid and itemid were original column names
        user_col_name : string
            userid column name
        item_col_name : string
            itemid column name
        feedback_col_name : string
            feedback column name

        Returns
        -------
        graph structure

        """
        # user_col_name = user_features.index.name
        # item_col_name = item_features.index.name

        if type(user_features) == pd.core.frame.DataFrame:
            user_features = user_features.values

        if type(item_features) == pd.core.frame.DataFrame:
            item_features = item_features.values

        graph = HeteroData()
        graph[user_col_name].x = torch.from_numpy(user_features).float()
        graph[item_col_name].x = torch.from_numpy(item_features).float()
        edge_index = torch.tensor(
            [
                dataset[user_col_name].astype(dtype="int64"),
                dataset[item_col_name].astype(dtype="int64"),
            ]
        ).to(torch.int64)
        rating = torch.from_numpy(dataset[feedback_col_name].values).to(torch.float)

        graph[user_col_name, "rates", item_col_name].edge_index = edge_index
        graph[user_col_name, "rates", item_col_name].edge_label = rating
        graph = transforms.ToUndirected()(graph)
        del graph[
            item_col_name, "rev_rates", user_col_name
        ].edge_label  # Remove "reverse" label.
        return graph

    def train_val_test_split(
        self, validation_frac=0.1, test_fraction=0.1, neg_sampling_ratio=0.0
    ):
        """
        Perform a link-level split into training, validation, and test edges.

        Parameters
        ----------
        validation_frac : float
        test_fraction : float
        neg_sampling_ratio : float

        """

        seed_everything(self.seed)
        self.train_data, self.val_data, self.test_data = transforms.RandomLinkSplit(
            num_val=validation_frac,
            num_test=test_fraction,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_types=[(self.__user_node_name, "rates", self.__item_node_name)],
            rev_edge_types=[
                (self.__item_node_name, "rev_rates", self.__user_node_name)
            ],
            is_undirected=self.graph.is_undirected(),
        )(self.graph)
        # return train_data, val_data, test_data

    def __train(self, train_graph, model, optimizer, loss):
        """Training function for model."""

        seed_everything(self.seed)
        model.train()
        optimizer.zero_grad()
        pred = model(
            train_graph.x_dict,
            train_graph.edge_index_dict,
            train_graph[self.__user_node_name, self.__item_node_name].edge_label_index,
        )
        target = train_graph[self.__user_node_name, self.__item_node_name].edge_label
        loss = loss(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss.sqrt())

    @torch.no_grad()
    def __test(self, graph, model, metric=functional.mse_loss):
        """Perform a model evaluation on test set."""
        seed_everything(self.seed)
        model.eval()
        pred = model(
            graph.x_dict,
            graph.edge_index_dict,
            graph[self.__user_node_name, self.__item_node_name].edge_label_index,
        )
        pred = pred.clamp(min=0, max=5)
        target = graph[self.__user_node_name, self.__item_node_name].edge_label.float()
        rmse = functional.mse_loss(pred, target).sqrt()
        return float(rmse)

    def fit(
        self,
        n_epochs=100,
        hidden_layers=16,
        learning_rate=0.01,
        e_patience=10,
        min_acc=0.05,
    ):
        """
        Initiate model training.

        Parameters
        ----------
        n_epochs : int
            number of epochs
        hidden_layers : int
            number of hidden layers
        learning_rate : float
            learning rate
        e_patience : int
            number of epochs to wait for early stopping
        min_acc : float
            minimum accuracy difference required for early stopping

        Returns
        -------
        pandas.DataFrame
            Results
        model
        """
        t0 = time.time()
        seed_everything(self.seed)
        model = Model(self.graph, hidden_layers)
        # Due to lazy initialization, we need to run one model step so the number
        # of parameters can be inferred:
        with torch.no_grad():
            model.encoder(self.train_data.x_dict, self.train_data.edge_index_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        k = 0
        loss, train_rmse, val_rmse, test_rmse = [], [], [], []
        for epoch in range(n_epochs):
            loss += [
                self.__train(
                    self.train_data, model, optimizer, loss=functional.mse_loss
                )
            ]

            train_rmse += [
                self.__test(self.train_data, model, metric=functional.mse_loss)
            ]

            val_rmse += [self.__test(self.val_data, model, metric=functional.mse_loss)]

            test_rmse += [
                self.__test(self.test_data, model, metric=functional.mse_loss)
            ]
            print(
                f"Epoch: {epoch+1:03d}, Loss: {loss[-1]:.4f}, Train: {train_rmse[-1]:.4f}, "
                f"Val: {val_rmse[-1]:.4f}, Test: {test_rmse[-1]:.4f}"
            )
            results = pd.DataFrame(
                {
                    "loss": loss,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "test_rmse": test_rmse,
                    "time": (time.time() - t0) / 60,
                }
            )

            # enable early stopping
            if (epoch > 1) and abs(loss[-1] / loss[-2] - 1) < min_acc:
                k += 1
            if k > e_patience:
                print("Early stopping")
                break
        self.model = model
        return results, model

    def __recommendation_logic(self, userid):
        """
        Predict feedback for non linked items to the user.

        Parameters
        ----------
        userid : list
            list of original user ids

        Returns
        -------
        pandas.DataFrame
            dataframe of items with recommendations

        """
        user_cols = self.__user_mapping.columns
        item_cols = self.__item_mapping.columns
        mapped_user_id = self.__user_mapping[
            self.__user_mapping[user_cols[1]].isin(userid)
        ][user_cols[0]].values
        already_watched = self.__edge_data[
            self.__edge_data[self.__node_types[0]].isin(mapped_user_id)
        ]
        not_watched = pd.DataFrame(
            list(
                itertools.product(
                    mapped_user_id, self.__item_mapping[self.__node_types[1]]
                )
            ),
            columns=self.__node_types,
        )

        not_watched = pd.concat(
            [
                not_watched,
                already_watched,
            ]
        ).drop_duplicates(self.__node_types, keep=False)
        edge_label_index = torch.tensor(
            [
                not_watched[self.__node_types[0]].values,
                not_watched[self.__node_types[1]].values,
            ]
        )
        with torch.no_grad():
            # test_data.to(device)
            seed_everything(self.seed)
            pred = self.model(
                self.graph.x_dict,
                self.graph.edge_index_dict,
                edge_label_index,
            )
            pred = pred.clamp(min=min(self.ratings), max=max(self.ratings)).numpy()
            if self.scale_rating:
                pred = self.__scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        not_watched[self.__feedback_col] = pred

        # return not_watched
        not_watched_2 = pd.merge(not_watched, self.__user_mapping, how="left")
        not_watched_2 = pd.merge(not_watched_2, self.__item_mapping, how="left").drop(
            columns=self.__node_types
        )
        return not_watched_2[[user_cols[-1], item_cols[-1], self.__feedback_col]]

    def recommend(self, userid, n_recommendations=10):
        """
        Recommend top k items for users.

        Parameters
        ----------
        userid : list
            list of original user ids
        n_recommendations : int
            top n recommendations

        Returns
        -------
        pandas.DataFrame
            dataframe of items with recommendations
        """
        user_cols = self.__user_mapping.columns
        item_cols = self.__item_mapping.columns
        recommendations = self.__recommendation_logic(userid=userid)
        top_k_index = (
            recommendations.groupby(user_cols[-1])[self.__feedback_col]
            .nlargest(n_recommendations)
            .to_frame()
            .reset_index()
        )
        top_k_recommendations = recommendations.loc[top_k_index["level_1"], :]
        top_k_recommendations = (
            top_k_recommendations.groupby(user_cols[-1])
            .apply(
                lambda x: dict(
                    x[[item_cols[-1], self.__feedback_col]].itertuples(index=False)
                )
            )
            .to_frame()
            .rename(columns={0: "recommendations"})
        )
        return top_k_recommendations
