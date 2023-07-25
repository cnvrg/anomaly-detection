# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

from tensorflow import keras
import os
import numpy as np
import random
from joblib import load
from sklearn import preprocessing
import argparse
import warnings

warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numba import njit
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_consistent_length
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import pandas as pd

clf2_path = "/input/comparexgbod/xgbod.joblib"
clf1_path = "/input/compare/clf"
datapath = "/input/compare/winner_details.csv"
miscolpath = "/input/preprocess/mis_col_type.csv"
origcolpath = "/input/preprocess/original_col.csv"

models = "/input/preprocess/"
columns_list_loc = models + "columns_list.csv"
scaler_loc = models + "std_scaler.bin"
label_encoder_loc = models + "ordinal_enc"
enc_loc = models + "encoded_values_file"
pca_model_loc = models + "pca_model"

original_col = pd.read_csv(origcolpath)
columns_list_1 = pd.read_csv(columns_list_loc)
id_column = columns_list_1["id_columns"].dropna().tolist()
label_encoding_columns = columns_list_1["label_encoded_columns"].dropna().tolist()
drop_list = id_column[0]
mis_col_type = pd.read_csv(miscolpath)
winner_details = pd.read_csv(datapath)
winner_model = winner_details["winner"].values[0]
neural_model = winner_model == "autoencoder" or winner_model == "deepsvdd"

########## loading models ############
label_encoder = None
enc = None
sc = None
if os.path.exists(label_encoder_loc):
    label_encoder = load(label_encoder_loc)
if os.path.exists(enc_loc):
    enc = load(enc_loc)
if os.path.exists(scaler_loc):
    sc = load(scaler_loc)
loaded_pca = None
if os.path.exists(pca_model_loc):
    loaded_pca = load(pca_model_loc)

clf1 = None
# load model from compare output
if neural_model:
    clf1 = keras.models.load_model(clf1_path)
    threshold_ = float(winner_details["threshold"].values[0])
else:
    clf1 = load(clf1_path + ".joblib")

# load model from xgbod output
clf2 = load(clf2_path)

# utility functions
def decision_function(X):
    """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
    X = check_array(X)

    X_norm = np.copy(X)

    # Predict on X and return the reconstruction errors
    pred_scores = clf1.predict(X_norm)
    if winner_model == "deepsvdd":
        return pred_scores
    return pairwise_distances_no_broadcast(X_norm, pred_scores)


def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.
    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples
    Y : array of shape (n_samples, n_features)
        Second input samples
    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError(
            "pairwise_distances_no_broadcast function receive"
            "matrix with different shapes {0} and {1}".format(X.shape, Y.shape)
        )
    return _pairwise_distances_no_broadcast_helper(X, Y)


@njit
def _pairwise_distances_no_broadcast_helper(X, Y):  # pragma: no cover
    """Internal function for calculating the distance with numba. Do not use.
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples
    Y : array of shape (n_samples, n_features)
        Second input samples
    Returns
    -------
    distance : array of shape (n_samples,)
        Intermediate results. Do not use.
    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def predict_model(X, return_confidence=False):
    """Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.
        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        confidence : numpy array of shape (n_samples,).
            Only if return_confidence is set to True.
        """

    pred_score = decision_function(X)
    prediction = (pred_score > threshold_).astype("int").ravel()

    if return_confidence:
        confidence = self.predict_confidence(X)
        return prediction, confidence

    return prediction


def predict(data):

    # data_new is the dataframe read from the csv
    # we need to convert input data into a form compatible with data_new
    if isinstance(data["vars"], str):
        data = data["vars"]
        data = data.split(",")
    else:
        data = data["vars"]

    data_new = pd.DataFrame([data])
    data_new.columns = original_col.columns[:-1]

    ##################################### Mising value treatment #####################################
    for colname, coltype in data_new.dtypes.iteritems():
        if (pd.isna(data_new[colname][0]) == True) or (data_new[colname][0] == ""):
            test = mis_col_type.isin([colname])
            for col in test:
                if test[col].sort_values(ascending=False).unique()[0] == True:
                    if col == "Mean":
                        data_new[colname][0] = original_col[colname][0]
                    elif col == "0-1":
                        data_new[colname][0] = random.choice([1, 0])
                    elif col == "Median":
                        data_new[colname][0] = original_col[colname][0]
                    elif col == "Yes-No":
                        data_new[colname][0] = random.choice(["Yes", "No"])
                    else:
                        data_new[colname][0] = original_col[colname][0]

    for col1, col2 in original_col.dtypes.iteritems():
        if col1 != "Class":
            data_new[col1] = data_new[col1].astype(col2)

    global label_encoding_columns
    if label_encoding_columns != ["None"]:
        garbage_index_0 = []
        for colname in label_encoding_columns:
            garbage_index = data_new.index[
                data_new[colname] == "Garbage-Value-999"
            ].tolist()
            garbage_index_0.append(garbage_index)

    ################################### defining binary map function #################################
    def binary_map(feature):
        return feature.map({"Yes": 1, "No": 0})

    ################################## label encoding ######################################
    cnt_garb = 0
    if label_encoder != None:
        label_encoding_columns = sorted(label_encoding_columns)
        data_new[label_encoding_columns] = label_encoder.transform(
            data_new[label_encoding_columns]
        )
        for colname in label_encoding_columns:
            bad_df = data_new.index.isin(garbage_index_0[cnt_garb])
            median_missing_label = data_new[~bad_df][colname].median().round()
            for j in range(len(garbage_index_0[cnt_garb])):
                data_new.at[
                    garbage_index_0[cnt_garb][j], colname
                ] = median_missing_label
            cnt_garb = cnt_garb + 1

    for colname, coltype in data_new.dtypes.iteritems():
        if "Garbage-Value-999" in colname:
            data_new = data_new.drop(colname, 1, errors="ignore")

    cat_var = columns_list_1["OHE_columns"].dropna().tolist()
    num_of_cat_var = len(cat_var)
    if enc != None:
        for colname, coltype in data_new.dtypes.iteritems():
            if (
                coltype == "object"
                and (colname not in id_column)
                and (colname not in cat_var)
                and (colname not in label_encoding_columns)
            ):
                data_new[colname] = pd.DataFrame(
                    binary_map(data_new[colname]), columns=[colname]
                )[colname]
    ################################### ONE HOT ENCODING #############################################
    if cat_var != []:
        cat_var = sorted(cat_var)
        newlist = []
        for i in range(len(enc.categories_)):
            for j in range(len(enc.categories_[i])):
                newlist.append(cat_var[i] + "-" + enc.categories_[i].tolist()[j])

        temp_ohe = pd.DataFrame(
            enc.transform(data_new[cat_var]).toarray().tolist(), columns=newlist
        )
        data_new = pd.concat([data_new, temp_ohe], axis=1)
        for colname in cat_var:
            data_new = data_new.drop([colname], axis=1)

    ######################################
    # load the normalizer
    percentage_cat_var = num_of_cat_var / data_new.shape[1]
    if sc != None:
        temp_df = pd.DataFrame()
        for colname, coltype in data_new.dtypes.iteritems():
            test1 = mis_col_type.isin([colname])
            for col in test1:
                if test1[col].sort_values(ascending=False).unique()[0] == True:
                    if col == "Mean":
                        temp_df[colname] = data_new[colname]

        temp_df = sc.transform(temp_df)

        for colname0 in temp_df:
            for colname1 in data_new:
                if colname0 == colname1:
                    data_new[colname1] = temp_df[colname0]
    ######################################

    ###################### drop id and dependent columns ##################
    if(drop_list!="None"):
        temp_col = data_new[[drop_list]]
        temp_df = data_new.drop(columns=[drop_list])
    else:
        temp_col=None
        temp_df = data_new

    ###################### drop id and dependent columns ##################

    ########### PCA ##############
    if loaded_pca != None:
        data_new_pca = loaded_pca.transform(temp_df)
        data_new_pca = pd.DataFrame(data_new_pca)
        print(data_new_pca.columns.tolist())
        if(temp_col is not None):
            data_new_pca = pd.concat(
                [data_new_pca, temp_col], axis=1
            )  # add id and dependent columns
    else:
        data_new_pca = data_new
    ########### PCA ##############

    columns = data_new_pca.columns.tolist()
    columns = [c for c in columns if c not in [id_column[0], "Class"]]
    X = data_new_pca[columns]

    # make prediction model 1

    if neural_model:
        p1 = predict_model(X)
    else:
        p1 = clf1.predict(X)
    # make prediction model 2
    p2 = clf2.predict(X)
    # perform OR operation

    pred = np.logical_or(p1, p2)
    pred = np.where(pred == True, 1, 0)

    response = {}
    try:
        response[str(data_new_pca[id_column[0]].values[0])] = str(pred[0])
    except:
        response["anomaly"] = str(pred[0])

    return response
