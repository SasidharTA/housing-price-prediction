# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""
import itertools
import logging
import pandas as pd
import time
from keras import initializers
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn import preprocessing

from .. import config as CONFIG
from .base import TimeSeriesModels

_LOGGER = logging.getLogger(__name__)


class LSTMSeqToSeqMultivariate(TimeSeriesModels):
    """Lstmsseqtosewmultivariate class."""

    def __init__(self):
        TimeSeriesModels.__init__(self)
        self.algo = "LSTMSeqToSeqMultivariate"
        self.hyperParams = self.createSearchSpace()

    #

    def createSearchSpace(self):
        """Creates search space for Exponentialsmoothingholtwinters class."""
        confDict = CONFIG.hyperparams_lstmseqtoseqmultivariate
        n_epochs = (
            confDict["n_epochs"] if (len(confDict["n_epochs"]) > 0) else [100, 200]
        )
        batch_size = (
            confDict["batch_size"] if (len(confDict["batch_size"]) > 0) else [10, 50]
        )
        n_hidden_layers = (
            confDict["n_hidden_layers"]
            if (len(confDict["n_hidden_layers"]) > 0)
            else [1, 2]
        )
        ip_seq_len = (
            confDict["ip_seq_len"] if (len(confDict["ip_seq_len"]) > 0) else [3, 5, 10]
        )
        ip_to_op_offset = (
            confDict["ip_to_op_offset"]
            if (len(confDict["ip_to_op_offset"]) > 0)
            else [10, 30]
        )
        op_seq_len = (
            confDict["op_seq_len"] if (len(confDict["op_seq_len"]) > 0) else [5, 10]
        )
        n_lstm_units_in_hidden_layers = (
            confDict["n_lstm_units_in_hidden_layers"]
            if (len(confDict["n_lstm_units_in_hidden_layers"]) > 0)
            else [5, 50]
        )
        n_lstm_units_decay_percent = (
            confDict["n_lstm_units_decay_percent"]
            if (len(confDict["n_lstm_units_decay_percent"]) > 0)
            else [40]
        )
        optimizer = (
            confDict["optimizer"]
            if (len(confDict["optimizer"]) > 0)
            else ["sgd", "adam"]
        )
        loss = (
            confDict["loss"]
            if (len(confDict["loss"]) > 0)
            else ["mean_squared_error", "mean_absolute_percentage_error"]
        )
        #
        paramLst = []
        for item in list(
            itertools.product(
                n_epochs,
                batch_size,
                n_hidden_layers,
                ip_seq_len,
                ip_to_op_offset,
                op_seq_len,
                n_lstm_units_in_hidden_layers,
                n_lstm_units_decay_percent,
                optimizer,
                loss,
            )
        ):
            paramLst.append(
                {
                    "n_epochs": item[0],
                    "batch_size": item[1],
                    "n_hidden_layers": item[2],
                    "ip_seq_len": item[3],
                    "ip_to_op_offset": item[4],
                    "op_seq_len": item[5],
                    "n_lstm_units_in_hidden_layers": item[6],
                    "n_lstm_units_decay_percent": item[7],
                    "optimizer": item[8],
                    "loss": item[9],
                }
            )
        return paramLst

    #

    def startModelProcess(self):
        """Starts model process for Exponentialsmoothingholtwinters class."""
        #
        _LOGGER.info("starting model : " + self.algo)
        st = time.time()
        global inpData
        inpData, trainDataDF, testDataDF = self.readData()
        #

        def getPredictions(
            model, trainDF, testDF, train_x, test_x, op_seq_len, batchSize
        ):
            """Gets predictions for Exponentialsmoothingholtwinters class."""
            #
            trainDFPredicted = trainDF.copy()
            testDFPredicted = testDF.copy()
            #
            model.save(CONFIG.code_output_path + self.algo + "_fitted_model.h5")
            # model = load_model(CONFIG.code_output_path + self.algo + "_fitted_model.h5")
            #
            predVars = [
                CONFIG.dv_variable_name + "_forecast_predicted_" + str(val)
                for val in range(1, op_seq_len + 1)
            ]
            #
            predTrain = model.predict(train_x, batch_size=batchSize)
            trainDFPredicted[predVars] = pd.DataFrame(predTrain, columns=predVars)
            #
            predTest = model.predict(test_x, batch_size=batchSize)
            testDFPredicted[predVars] = pd.DataFrame(predTest, columns=predVars)
            #
            return trainDFPredicted, testDFPredicted, predVars

        def get_transformed_data(given_df, method="MinMax"):
            """Gets transformed data for Exponentialsmoothingholtwinters class."""
            if method == "MinMax":
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                scaled_df = pd.DataFrame(scaler.fit_transform(given_df))
                scaled_df.columns = given_df.columns
                return scaled_df, scaler
            else:
                _LOGGER.info("returning without transforming the data!")
                return given_df

        def runModel(paramDict):
            """Runs model for Exponentialsmoothingholtwinters class."""
            inpDataDF = inpData.copy()
            # prepare input sequence
            ipSeqVars = []
            for col in CONFIG.idv_variable_names:
                for val in range(1, paramDict["ip_seq_len"] + 1):
                    newVar = str(val) + "_lag_" + col
                    ipSeqVars.append(newVar)
                    inpDataDF[newVar] = inpDataDF[col].shift(val)
            # prepare output sequence
            opSeqVars = []
            for val in range(1, paramDict["op_seq_len"] + 1):
                newVar = str(val) + "_forecast_actual_" + CONFIG.dv_variable_name
                opSeqVars.append(newVar)
                inpDataDF[newVar] = inpDataDF[CONFIG.dv_variable_name].shift(
                    -1 * (paramDict["ip_to_op_offset"] + val)
                )
            # split data into train and test
            inpDataDF = inpDataDF.dropna()
            inpDataDF = inpDataDF.sort_values("Date", ascending=True)
            # scale complete data (train + test)
            # inpDataDF, scaler_X = get_transformed_data(inpDataDF[ipSeqVars+opSeqVars])
            inpXDF = inpDataDF.loc[:, ipSeqVars].reindex_axis(
                sorted(
                    inpDataDF[ipSeqVars].columns, key=lambda x: int(x[0 : x.find("_")])
                ),
                axis=1,
            )

            temp_cols = inpXDF.columns
            inpXDF = inpXDF[inpXDF.columns[::-1]]
            inpXDF.columns = temp_cols

            inpYDF = inpDataDF.loc[:, opSeqVars].reindex_axis(
                sorted(
                    inpDataDF[opSeqVars].columns, key=lambda x: int(x[0 : x.find("_")])
                ),
                axis=1,
            )
            splitInd = int(len(inpXDF) * 0.80)
            trainDF = pd.concat(
                [
                    inpXDF[0:splitInd].reset_index(drop=True),
                    inpYDF[0:splitInd].reset_index(drop=True),
                ],
                axis=1,
            )
            testDF = pd.concat(
                [
                    inpXDF[splitInd:].reset_index(drop=True),
                    inpYDF[splitInd:].reset_index(drop=True),
                ],
                axis=1,
            )
            trainDF, scaler_X = get_transformed_data(trainDF[ipSeqVars + opSeqVars])
            testDF = pd.DataFrame(
                scaler_X.transform(testDF[ipSeqVars + opSeqVars]),
                columns=ipSeqVars + opSeqVars,
            )

            ipSeqVarsSorted = inpXDF.columns.tolist()
            opSeqVarsSorted = inpYDF.columns.tolist()

            # adjust sample size - keras requires number of_samples to be divisible by batch size
            def adjustSampleSize(df):
                """Adjusts sample size for Exponentialsmoothingholtwinters class."""
                while 1 == 1:
                    if len(df) % paramDict["batch_size"] != 0:
                        df = df[0 : len(df) - 1]
                    else:
                        break
                return df

            trainDF = adjustSampleSize(trainDF)
            testDF = adjustSampleSize(testDF)

            trainDFScaled = trainDF
            testDFScaled = testDF

            train_x = trainDFScaled[ipSeqVarsSorted].values.reshape(
                len(trainDFScaled),
                paramDict["ip_seq_len"],
                len(CONFIG.idv_variable_names),
            )
            train_y = trainDF[opSeqVarsSorted].values.reshape(
                len(trainDF), paramDict["op_seq_len"]
            )
            test_x = testDFScaled[ipSeqVarsSorted].values.reshape(
                len(testDFScaled),
                paramDict["ip_seq_len"],
                len(CONFIG.idv_variable_names),
            )
            test_y = testDF[opSeqVarsSorted].values.reshape(len(testDF), paramDict['op_seq_len'])  # noqa
            #
            # create LSTM network architecture based on configurations
            model = Sequential()
            n_hidden_layers = paramDict["n_hidden_layers"]
            if n_hidden_layers == 1:
                model.add(
                    LSTM(
                        paramDict["n_lstm_units_in_hidden_layers"],
                        batch_input_shape=(
                            paramDict["batch_size"],
                            train_x.shape[1],
                            train_x.shape[2],
                        ),
                        stateful=True,
                        kernel_initializer=initializers.RandomNormal(
                            mean=0, stddev=0.05
                        ),
                        recurrent_initializer=initializers.RandomNormal(
                            mean=0, stddev=0.05
                        ),
                    )
                )
            else:
                n_lstm_units = paramDict["n_lstm_units_in_hidden_layers"]
                for hlayer in range(1, n_hidden_layers):
                    model.add(
                        LSTM(
                            n_lstm_units,
                            batch_input_shape=(
                                paramDict["batch_size"],
                                train_x.shape[1],
                                train_x.shape[2],
                            ),
                            stateful=True,
                            kernel_initializer=initializers.RandomNormal(
                                mean=0, stddev=0.05
                            ),
                            recurrent_initializer=initializers.RandomNormal(
                                mean=0, stddev=0.05
                            ),
                            return_sequences=True,
                        )
                    )
                    n_lstm_units = n_lstm_units - round(
                        (paramDict["n_lstm_units_decay_percent"] / 100) * n_lstm_units
                    )
                    n_lstm_units = n_lstm_units if n_lstm_units > 1 else 2
                model.add(
                    LSTM(
                        n_lstm_units,
                        batch_input_shape=(
                            paramDict["batch_size"],
                            train_x.shape[1],
                            train_x.shape[2],
                        ),
                        stateful=True,
                        kernel_initializer=initializers.RandomNormal(
                            mean=0, stddev=0.05
                        ),
                        recurrent_initializer=initializers.RandomNormal(
                            mean=0, stddev=0.05
                        ),
                    )
                )
            model.add(Dense(train_y.shape[1]))
            model.compile(loss=paramDict["loss"], optimizer=paramDict["optimizer"])
            # run epochs
            for i in range(paramDict["n_epochs"]):
                model.fit(
                    train_x,
                    train_y,
                    epochs=1,
                    batch_size=paramDict["batch_size"],
                    verbose=0,
                    shuffle=False,
                )
                model.reset_states()
                # _LOGGER.info("----------------- completed epochs : " + str(i))
            trainDFPredicted, testDFPredicted, predVars = getPredictions(
                model,
                trainDF,
                testDF,
                train_x,
                test_x,
                paramDict["op_seq_len"],
                paramDict["batch_size"],
            )

            actual_output = pd.DataFrame(
                scaler_X.inverse_transform(
                    trainDFPredicted[ipSeqVarsSorted + opSeqVarsSorted]
                ),
                columns=ipSeqVarsSorted + opSeqVarsSorted,
            )
            predicted_output = pd.DataFrame(
                scaler_X.inverse_transform(
                    trainDFPredicted[ipSeqVarsSorted + predVars]
                ),
                columns=ipSeqVarsSorted + predVars,
            )
            actual_output[predVars] = predicted_output[predVars]
            actual_output = actual_output.applymap(float)
            trainDFPredicted = actual_output.copy()
            del actual_output
            actual_output = pd.DataFrame(
                scaler_X.inverse_transform(
                    testDFPredicted[ipSeqVarsSorted + opSeqVarsSorted]
                ),
                columns=ipSeqVarsSorted + opSeqVarsSorted,
            )
            predicted_output = pd.DataFrame(
                scaler_X.inverse_transform(testDFPredicted[ipSeqVarsSorted + predVars]),
                columns=ipSeqVarsSorted + predVars,
            )
            actual_output[predVars] = predicted_output[predVars]
            actual_output = actual_output.applymap(float)
            testDFPredicted = actual_output
            temp_list = []
            for i in range(1, paramDict["op_seq_len"] + 1):
                (
                    rmse_train,
                    mape_train,
                    mae_train,
                    rsqTrain,
                ) = self.getAccuracyMetricsBase(
                    trainDFPredicted[opSeqVarsSorted[i - 1]],
                    trainDFPredicted[predVars[i - 1]],
                )
                (
                    rmse_test,
                    mape_test,
                    mae_test,
                    rsqTestDummy,
                ) = self.getAccuracyMetricsBase(
                    testDFPredicted[opSeqVarsSorted[i - 1]],
                    testDFPredicted[predVars[i - 1]],
                )
                temp_var = self.algo + "@forecast_sequence_" + str(i)
                temp_list.append(
                    pd.DataFrame(
                        [
                            [
                                temp_var,
                                paramDict,
                                rmse_train,
                                mape_train,
                                mae_train,
                                rsqTrain,
                                rmse_test,
                                mape_test,
                                mae_test,
                            ]
                        ]
                    )
                )

            temp_df = pd.concat(temp_list, axis=0)
            temp_df.columns = [
                "algo",
                "hyperParams",
                "rmse_train",
                "mape_train",
                "mae_train",
                "rsqTrain",
                "rmse_test",
                "mape_test",
                "mae_test",
            ]
            return temp_df

        resultLst = []
        for val in self.hyperParams:
            resultLst.append(runModel(val))
        _LOGGER.info("time taken :  %f minutes" % (((time.time() - st) / (60.0))))

        return pd.concat(resultLst)
