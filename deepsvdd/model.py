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

from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    average_precision_score,
)
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from cnvrg import Experiment
import os

e = Experiment()

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

parser = argparse.ArgumentParser(description="""Creator""")
parser.add_argument(
    "-d",
    "--data_dir",
    action="store",
    dest="data_dir",
    default="/data/anomalydata/creditcard.csv",
    required=False,
    help="""path to anomaly data""",
)
#########hyperparams for optimization#########
parser.add_argument(
    "--hidden_neurons",
    action="store",
    dest="hidden_neurons",
    default="5",
    required=False,
    help="""Hidden layers structure""",
)

parser.add_argument(
    "--epochs",
    action="store",
    dest="epochs",
    default="100",
    required=False,
    help="""training epochs""",
)
parser.add_argument(
    "--dropout_rate",
    action="store",
    dest="dropout_rate",
    default="0.2",
    required=False,
    help="""droupout used in hidden layers""",
)
parser.add_argument(
    "--l2_regularizer",
    action="store",
    dest="l2_regularizer",
    default="0.1",
    required=False,
    help="""value of l2 regularizer""",
)
parser.add_argument(
    "--use_ae",
    action="store",
    dest="use_ae",
    default="False",
    required=False,
    type=bool,
    help="""use as autoencoder""",
)


args = parser.parse_args()
datapath = args.data_dir
epochs = int(args.epochs)
dropout_rate = float(args.dropout_rate)
l2_regularizer = float(args.l2_regularizer)
hidden_neurons = args.hidden_neurons
use_ae = args.use_ae
args = parser.parse_args()
datapath = args.data_dir
data1 = pd.read_csv(datapath)
idcol = pd.read_csv("/input/preprocess/columns_list.csv")
id_column = idcol["id_columns"].dropna().tolist()

columns = data1.columns.tolist()
# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class", id_column[0]]]
# Store the variable we are predicting
target = "Class"
# Define a random state
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)

trainX, testX, trainy, testy = train_test_split(
    X, Y, test_size=0.2, random_state=2, stratify=Y
)
Fraud = trainy[trainy == 1]
Valid = trainy[trainy == 0]
outlier_fraction = len(Fraud) / float(len(trainy))


random_state = np.random.RandomState(42)
if args.hidden_neurons != "default":
    # define the neural network architecture based on the input args
    hidden_neurons = int(args.hidden_neurons)
    hidden_neurons = [x for x in range(5, 1, -1)]
    hidden_neurons.extend([0.5])
    hidden_neurons = np.array(hidden_neurons)
    hidden_neurons = list(hidden_neurons * trainX.shape[1])
    hidden_neurons[-1] = max(trainX.shape[1] // 2, 1)

    classifiers = {
        "DeepSVDD": DeepSVDD(
            hidden_neurons=hidden_neurons,
            epochs=epochs,
            dropout_rate=dropout_rate,
            l2_regularizer=l2_regularizer,
            contamination=outlier_fraction,
            use_ae=use_ae,
            preprocessing=False,
        )
    }
else:
    hidden_neurons = [64, 32, max(trainX.shape[1] // 2, 1)]
    classifiers = {
        "DeepSVDD": DeepSVDD(
            hidden_neurons=hidden_neurons,
            epochs=epochs,
            dropout_rate=dropout_rate,
            l2_regularizer=l2_regularizer,
            contamination=outlier_fraction,
            use_ae=use_ae,
            preprocessing=False,
        )
    }

for i, (clf_name, clf) in enumerate(classifiers.items()):
    print("Now fitting: ", clf_name)
    clf.fit(trainX)
    y_pred = clf.predict(testX)
    n_errors = (y_pred != testy).sum()
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score :")
    print(accuracy_score(testy, y_pred))
    print("Classification Report :")
    print(classification_report(testy, y_pred))
    print("roc_score: ", roc_auc_score(testy, clf.decision_function(testX)))
    e.log_param(clf_name, roc_auc_score(testy, clf.decision_function(testX)))
    print(average_precision_score(testy, y_pred))
    e.log_param("average_precision_score", average_precision_score(testy, y_pred))
    e.log_param("threshold", clf.threshold_)
    # save the trained model
    clf.model_.save(cnvrg_workdir + "/clf")
