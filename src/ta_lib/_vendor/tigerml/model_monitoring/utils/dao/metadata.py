import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import DateTime, Integer, String, bindparam
from sqlalchemy.sql import text
from tigerml.model_monitoring.utils.dao import db_connection


class MetaData:
    """
    MetaData class is part of dao (data access object) module.

    This class takes care of setting up metadata table for the first run,
    generating new metadata for the current run and adding metadata after genration of new metadata.

    Parameters
    ----------
    db_conn_string: str
        SQLAlchemy connection string which assumes the format as such:
        dialect+driver://username:password@host:port/database.
        To know more about this string refer - https://docs.sqlalchemy.org/en/14/core/engines.html.

    features: list[str]
        A list of features on which the model was trained on
    yhat: str
        Predicted target column name for data
    y: str
        Actual target column name for dat
    model_name: str
        Name of the Model for which you are performing the monitoring
    model_version: str
        Version of the model for which you are performing the monitoring
    """

    def __init__(
        self,
        db_conn_string,
        features,
        yhat,
        y,
        model_name,
        model_version,
    ):
        # TODO: Can we make user give their own Metadata apart from model_name & model_version?
        self.engine = db_connection.get_db_engine(connection_string=db_conn_string)
        self.features = features
        self.yhat_base = yhat
        self.yhat_curr = yhat
        self.y_base = y
        self.y_curr = y
        self.model_name = model_name
        self.model_version = model_version

    def setup_metadate(self):
        """
        This function sets up metadata & model_features_metadata table for performing model_monitoring over time.

        You can call this function whenever you need to keep track of which model ran at what time.
        It also tracks how many times was the model monitoring run previously for the current model.

        This function returns ModelExists,RunID,TimeStamp parameters (Metadata for this model).
        These can help you keep track of your model_monitoring metrics at model and run level.

        Returns
        -------
        ModelExists: bool
            This is a flag which tells wheather the current model exists in the metadata table.
            If it is False, it indicates that the model_monitoring was performed for the first time on this model.
        RunID: int
            This tells how many times was the model_monitoring performed for the current model.
        TimeStamp: UTC timestamp
            This is the UTC timestamp of when the model_monitoring for the current model started.
        """
        # create DB in SQLLite Server, if doesn't exist
        # TODO: Make metadata_exists_query DB Agnostic. Rightnow it is specific to SQLite DB.
        with self.engine.connect() as db_conn:
            metadata_exists_query = text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
            )
            metadata_exists = db_conn.execute(metadata_exists_query).fetchone()
        if metadata_exists:
            (
                ModelExists,
                RunID,
                TimeStamp,
            ) = self._generate_metadata_for_run()
            self._add_metadata(
                ModelExists=ModelExists,
                RunID=RunID,
                TimeStamp=TimeStamp,
            )
        else:
            # Since metadata table is not there, we will CREATE metadata table and add the below metadata columns along with ModelInputs and ModelOutputs
            ModelExists = False
            RunID = 1
            TimeStamp = datetime.now(tz=timezone.utc)
            self._add_metadata(
                ModelExists=ModelExists,
                RunID=RunID,
                TimeStamp=TimeStamp,
            )
        return ModelExists, RunID, TimeStamp

    def _add_metadata(self, ModelExists, RunID, TimeStamp):
        row = {
            "ModelName": [self.model_name],
            "ModelVersion": [self.model_version],
            "RunID": [RunID],
            "TimeStamp": [TimeStamp],
        }
        metadata_df = pd.DataFrame(row)
        metadata_df.to_sql(
            name="metadata",
            con=self.engine,
            dtype={
                "ModelName": String(),
                "ModelVersion": String(),
                "RunID": Integer(),
                "TimeStamp": DateTime(),
            },
            index=False,
            if_exists="append",
        )
        if ModelExists is False:
            feature_name_list = (
                self.features
                + [self.yhat_base]
                + [self.yhat_curr]
                + [self.y_base]
                + [self.y_curr]
            )
            feature_type_list = (
                ["feature"] * len(self.features)
                + ["yhat_base"]
                + ["yhat_curr"]
                + ["y_base"]
                + ["y_curr"]
            )
            model_features_metadata_df = pd.DataFrame(
                columns=["ModelName", "ModelVersion", "FeatureType", "FeatureName"]
            )
            model_features_metadata_df["FeatureType"] = feature_type_list
            model_features_metadata_df["FeatureName"] = feature_name_list
            model_features_metadata_df["ModelName"] = self.model_name
            model_features_metadata_df["ModelVersion"] = self.model_version
            model_features_metadata_df.to_sql(
                name="metadata_model_features",
                con=self.engine,
                dtype={
                    "ModelName": String(),
                    "ModelVersion": String(),
                    "FeatureType": String(),
                    "FeatureName": String(),
                },
                index=False,
                if_exists="append",
            )

    def _generate_metadata_for_run(self):
        with self.engine.connect() as db_conn:
            model_exists_query = text(
                "SELECT RunID from metadata where ModelName= :ModelName AND ModelVersion= :ModelVersion ORDER BY RunID DESC"
            )
            model_exists_result = db_conn.execute(
                model_exists_query,
                ModelName=self.model_name,
                ModelVersion=self.model_version,
            ).fetchone()

        if model_exists_result:
            ModelExists = True
            LastRunID = model_exists_result[0]
            RunID = LastRunID + 1
            TimeStamp = datetime.now(tz=timezone.utc)
        else:
            ModelExists = False
            RunID = 1
            TimeStamp = datetime.now(tz=timezone.utc)

        return ModelExists, RunID, TimeStamp

    def _generate_metadata_for_run_using_features(self):
        with self.engine.connect() as db_conn:
            model_exists_query = text(
                """
SELECT DISTINCT m.ModelID, m.RunID,
CASE
    WHEN ((mfm.FeatureType == "feature") AND (mfm.FeatureName IN :features)) THEN "MATCH"
    WHEN (mfm.FeatureType in ("yhat_base","yhat_curr") AND mfm.FeatureName == :yhat_base) THEN "MATCH"
    WHEN (mfm.FeatureType in ("y_base","y_curr") AND (mfm.FeatureName == :y_base OR mfm.FeatureName is :y_base)) THEN "MATCH"
    ELSE "NOMATCH"
END AS ModelMatch
from metadata m join model_features_metadata mfm on m.ModelID == mfm.ModelID
ORDER BY m.ModelID,m.RunID DESC
"""
            )
            params = {
                "features": self.features,
                "yhat_base": self.yhat_base,
                "y_base": self.y_base,
            }
            model_exists_query = model_exists_query.bindparams(
                bindparam("features", expanding=True),
                bindparam("yhat_base"),
                bindparam("y_base"),
            )
            model_exists_result = db_conn.execute(model_exists_query, params).fetchall()

        all_fetched_models = []
        no_match_models = []
        for row in model_exists_result:
            ModelID = row[0]
            LastRunID = row[1]
            MatchFlag = row[2]
            all_fetched_models.append((ModelID, LastRunID))
            if MatchFlag == "NOMATCH":
                no_match_models.append((ModelID, LastRunID))
        all_fetched_models = set(all_fetched_models)
        no_match_models = set(no_match_models)
        matched_models = list(all_fetched_models - no_match_models)
        matched_models.sort(reverse=True)
        if matched_models:
            ModelExists = True
            ModelID = matched_models[0][0]
            LastRunID = matched_models[0][1]
            RunID = LastRunID + 1
            TimeStamp = datetime.now(tz=timezone.utc)
        else:
            with self.engine.connect() as db_conn:
                LastModelID_query = text(
                    f"SELECT ModelID FROM metadata ORDER BY ModelID DESC"
                )
                LastModelID = db_conn.execute(LastModelID_query).fetchone()
            ModelExists = False
            ModelID = LastModelID[0] + 1
            RunID = 1
            TimeStamp = datetime.now(tz=timezone.utc)
        return ModelExists, RunID, TimeStamp
