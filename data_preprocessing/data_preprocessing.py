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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import random
from joblib import dump, load
import joblib
from sklearn import preprocessing
import argparse
import warnings

warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
import os

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument(
    "-f",
    "--anomaly_data",
    action="store",
    dest="anomaly_data",
    default="/data/anomalydata/churn.csv",
    required=True,
    help="""churn data""",
)
parser.add_argument(
    "--label_encoding",
    action="store",
    dest="label_encoding",
    default="None",
    required=False,
    help="""label encoding columns""",
)
parser.add_argument(
    "--scaler",
    action="store",
    dest="scaler",
    default="minmax",
    required=False,
    help="""which scaler to use""",
)
parser.add_argument(
    "--id_column",
    action="store",
    dest="id_column",
    default="None",
    required=False,
    help="""id column""",
)
parser.add_argument(
    "--pca_arg",
    action="store",
    dest="pca_arg",
    default="0.95",
    required=False,
    help="""pca argument (if its percentage or number of components)""",
)
parser.add_argument(
    "--pca_overwrite",
    action="store",
    dest="pca_overwrite",
    default="False",
    required=False,
    help="""whether to ovwewrite pca argumrnt or not""",
)

args = parser.parse_args()
anomaly_data = args.anomaly_data
label_encoding_cols = sorted(args.label_encoding.split(","))
print(label_encoding_cols)
scaler = args.scaler
data_df = pd.read_csv(anomaly_data)

label_encoder = preprocessing.OrdinalEncoder()
id_columns = args.id_column
print(id_columns)
dependent_column = "Class"
print(dependent_column)
pca_arg = float(args.pca_arg)
print(pca_arg)
if id_columns == "None":
    drop_list = [dependent_column]
else:
    drop_list = [id_columns, dependent_column]
print(drop_list)
pca_overwrite = str(args.pca_overwrite)


#function to help get an overview of the csv file passed as input
def dataoveriew(df, message):
    print(f"{message}:\n")
    print("Number of rows: ", df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())


no_of_rows = data_df.shape[0]
mis_val_cnt = data_df.isnull().sum().values.sum()
no_of_features = data_df.shape[1]
dataoveriew(data_df, "Overview of the dataset")
mis_col_type = pd.DataFrame(columns=["Mean", "0-1", "Median", "Yes-No", "String"])

############################# Mising value treatment ##############################################
original_col = data_df.head(1)
cnt_col = 0
for colname, coltype in data_df.dtypes.iteritems():

    percentage_unique = len(data_df[colname].dropna().unique()) / len(
        data_df[colname].dropna()
    )
    first_value = data_df[colname].dropna().sort_values(ascending=True).unique()[0]
    length_unique = len(data_df[colname].dropna().unique())
    max_unique = max(data_df[colname].dropna().unique())

    if length_unique == 1:
        data_df.drop(colname, axis=1)
    elif ((coltype == "int64") or (coltype == "float64")) and (
        percentage_unique >= 0.1
    ):
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = np.mean(data_df[colname])
        mis_col_type.at[cnt_col, "Mean"] = colname
        cnt_col = cnt_col + 1
        original_col[colname][0] = np.mean(data_df[colname])
    elif (
        ((coltype == "int64") or (coltype == "float64"))
        and (max_unique < 2)
        and (length_unique <= 2)
    ):
        replacement = random.choices([0, 1], k=data_df[colname].isna().sum())
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = replacement
        mis_col_type.at[cnt_col, "0-1"] = colname
        cnt_col = cnt_col + 1
        original_col[colname][0] = 0
    elif ((coltype == "int64") or (coltype == "float64")) and (percentage_unique < 0.1):
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = np.median(data_df[colname])
        mis_col_type.at[cnt_col, "Median"] = colname
        cnt_col = cnt_col + 1
        original_col[colname][0] = np.median(data_df[colname])
    elif ((first_value.lower() == "no") or (first_value.lower() == "yes")) and (
        length_unique <= 2
    ):
        replacement = random.choices(["Yes", "No"], k=data_df[colname].isna().sum())
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = replacement
        mis_col_type.at[cnt_col, "Yes-No"] = colname
        cnt_col = cnt_col + 1
        original_col[colname][0] = "Yes"
    else:
        nans = data_df[colname].isna()
        data_df.loc[nans, colname] = "Garbage-Value-999"
        mis_col_type.at[cnt_col, "String"] = colname
        cnt_col = cnt_col + 1
        original_col[colname][0] = random.choice(data_df[colname].unique())

#################################### Removing label encoded missing values #######################
original_col.to_csv(cnvrg_workdir + "/original_col.csv", index=False)
garbage_index_0 = []
if label_encoding_cols != ["None"]:
    for colname in label_encoding_cols:
        garbage_index = data_df.index[data_df[colname] == "Garbage-Value-999"].tolist()
        garbage_index_0.append(garbage_index)

mis_col_type.to_csv(cnvrg_workdir + "/mis_col_type.csv", index=False)

################ Defining Binary Map function and mapping Churn column ##########################
def binary_map(feature):
    return feature.map({"Yes": 1, "No": 0})


if data_df[dependent_column].iloc[0] != 0 and data_df[dependent_column].iloc[0] != 1:
    data_df[dependent_column] = pd.DataFrame(
        binary_map(data_df[dependent_column]), columns=[dependent_column]
    )[dependent_column]

###################################### sparsity ##################################################
sparsity = 0
total = 0
for colname, coltype in data_df.dtypes.iteritems():
    if (coltype == "int64") or (coltype == "float64"):
        total = total + data_df[colname].shape[0]

sparsity = (data_df == 0).astype(int).sum(axis=1).sum() / total


######################################## one hot encoding = ######################################
num_of_cat_var = 0
cat_var = []
if (
    (scaler.lower() == "minmax")
    or (scaler.lower() == "min max")
    or (scaler.lower() == "min-max")
    or (scaler.lower() == "min_max")
):
    sc = MinMaxScaler()
else:
    sc = StandardScaler()

for colname, coltype in data_df.dtypes.iteritems():
    if coltype == "object" and colname not in id_columns.split(","):
        len_unique = len(data_df[colname].unique())
        first_val = data_df[colname].sort_values().unique()[0].lower()
        first_col = data_df[colname]
        if (
            (len_unique >= 2)
            and (colname not in label_encoding_cols)
            and not (len_unique == 2 and (first_val == "no" or first_val == "yes"))
        ):
            num_of_cat_var = num_of_cat_var + 1
            cat_var.append(colname)
        elif (
            (len_unique == 2)
            and (colname not in label_encoding_cols)
            and (first_val == "no" or first_val == "yes")
        ):
            data_df[colname] = pd.DataFrame(binary_map(first_col), columns=[colname])[
                colname
            ]
        else:
            print(colname)
            
##################################### using ordinal encoder to label encode ######################
cnt_garb = 0
if label_encoding_cols != ["None"]:
    label_encoder.fit(data_df[label_encoding_cols])
    dump(label_encoder, cnvrg_workdir + "/ordinal_enc")
    data_df[label_encoding_cols] = label_encoder.transform(data_df[label_encoding_cols])
    for colname in label_encoding_cols:
        bad_df = data_df.index.isin(garbage_index_0[cnt_garb])
        median_missing_label = data_df[~bad_df][colname].median().round()
        for j in range(len(garbage_index_0[cnt_garb])):
            data_df.at[garbage_index_0[cnt_garb][j], colname] = median_missing_label
        cnt_garb = cnt_garb + 1

percentage_cat_var = num_of_cat_var / data_df.shape[1]

###############################scaling the data##################3
if scaler != "None":
    temp_df = pd.DataFrame()
    for colname, coltype in data_df.dtypes.iteritems():
        if (len(data_df[colname].unique()) / len(data_df[colname]) > 0.1) and (
            coltype == "int64" or coltype == "float64"
        ):
            temp_df[colname] = data_df[colname]

    temp_df = sc.fit_transform(temp_df)
    for colname0 in temp_df:
        for colname1 in data_df:
            if colname0 == colname1:
                data_df[colname1] = temp_df[colname0]

    dump(sc, cnvrg_workdir + "/std_scaler.bin", compress=True)

#################################### one hot encoding function ###################################
def get_encoder_inst(feature_col):
    """
    returns: an instance of sklearn OneHotEncoder fit against a (training) column feature;
    such instance is saved and can then be loaded to transform unseen data
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(data_df[feature_col])
    ohe_column_names = []
    for i in range(len(feature_col)):
        ohe_column_names.extend((feature_col[i] + "-" + enc.categories_[i]).tolist())
    fnames = ohe_column_names
    pframe = enc.transform(data_df[feature_col]).toarray().tolist()
    OHE_df = pd.DataFrame(pframe, columns=fnames)
    file_name = cnvrg_workdir + "/encoded_values_file"
    dump(enc, file_name)
    return OHE_df


if len(cat_var) != 0:
    cat_var = sorted(cat_var)
    OHE_df = get_encoder_inst(cat_var)
    data_df = pd.concat([data_df, OHE_df], axis=1)
    for colname in cat_var:
        data_df = data_df.drop([colname], axis=1)
    for colname, coltype in data_df.dtypes.iteritems():
        if "Garbage-Value-999" in colname:
            data_df = data_df.drop(colname, 1, errors="ignore")

data_df.to_csv(cnvrg_workdir + "/data_df.csv", index=False)
dimensionality_ratio = data_df.shape[0] / data_df.shape[1]
print(data_df.shape)

###################### drop id and dependent columns ##################
temp_col = data_df[drop_list]
print(temp_col.shape)

temp_df = data_df.drop(columns=drop_list)
temp_df.to_csv(cnvrg_workdir + "/temp_df.csv", index=False)


###################### PCA ##################
from sklearn.decomposition import PCA

no_of_features = temp_df.shape[1]
if no_of_features > 10:
    dopca = True
else:
    dopca = False

print(pca_overwrite)
if pca_overwrite == "True":
    dopca = not dopca
else:
    pass
if dopca:
    pca = PCA(pca_arg)
    data_df_pca = pca.fit_transform(temp_df)
    file_name = cnvrg_workdir + "/pca_model"
    dump(pca, file_name)
    print(data_df_pca.shape)
    data_df_pca = pd.DataFrame(data_df_pca)
    print(data_df_pca.columns.tolist())
else:
    data_df_pca = temp_df
data_df_pca = pd.concat([data_df_pca, temp_col], axis=1)  # add id and dependent columns

data_df_pca.to_csv(cnvrg_workdir + "/data_df_pca.csv", index=False)
data_df_pca.head(1).to_csv(cnvrg_workdir + "/processed_col.csv", index=False)
print(data_df_pca.shape)

#######################################no_of_features ############################################
dimensionality_ratio_pca = data_df_pca.shape[0] / data_df_pca.shape[1]

################################### compiling list of col types ####################
if len(id_columns.split(",")) > len(label_encoding_cols) and len(
    id_columns.split(",")
) > len(cat_var):
    columns_list = pd.DataFrame(columns=["id_columns"])
    columns_list["id_columns"] = id_columns.split(",")
    columns_list = pd.concat(
        [
            columns_list,
            pd.DataFrame(label_encoding_cols, columns=["label_encoded_columns"]),
        ],
        axis=1,
    )
    columns_list = pd.concat(
        [columns_list, pd.DataFrame(cat_var, columns=["OHE_columns"])]
    )
elif len(label_encoding_cols) > len(cat_var):
    columns_list = pd.DataFrame(columns=["label_encoded_columns"])
    columns_list["label_encoded_columns"] = label_encoding_cols
    columns_list = pd.concat(
        [columns_list, pd.DataFrame(id_columns.split(","), columns=["id_columns"])],
        axis=1,
    )
    columns_list = pd.concat(
        [columns_list, pd.DataFrame(cat_var, columns=["OHE_columns"])], axis=1
    )
else:
    columns_list = pd.DataFrame(columns=["OHE_columns"])
    columns_list["OHE_columns"] = cat_var
    columns_list = pd.concat(
        [columns_list, pd.DataFrame(id_columns.split(","), columns=["id_columns"])],
        axis=1,
    )
    columns_list = pd.concat(
        [
            columns_list,
            pd.DataFrame(label_encoding_cols, columns=["label_encoded_columns"]),
        ],
        axis=1,
    )

columns_list.to_csv(cnvrg_workdir + "/columns_list.csv", index=False)

################################ Logging metrics #################################################
from cnvrg import Experiment

e = Experiment()
e.log_param("sparsity", sparsity)
e.log_param("dimensionality_ratio", dimensionality_ratio)
e.log_param("dimensionality_ratio_pca", dimensionality_ratio_pca)
e.log_param("Categorical_Percentage", percentage_cat_var)
e.log_param("Number-of-rows", no_of_rows)
e.log_param("Missing value count", mis_val_cnt)
e.log_param("Number of features", no_of_features)
