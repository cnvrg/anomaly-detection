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

import os
from pyod.models.hbos import HBOS
from joblib import dump
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

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

e = Experiment()

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
# hyperparams
parser.add_argument(
    "--n_bins",
    action="store",
    dest="n_bins",
    default="10",
    required=False,
    help="""The number of bins. “auto” uses the birge-rozenblac method for automatic selection of the optimal number of bins for each feature.
""",
)
parser.add_argument(
    "--alpha",
    action="store",
    dest="alpha",
    default="0.1",
    required=False,
    help="""The regularizer for preventing overflow.
""",
)
parser.add_argument(
    "--tol",
    action="store",
    dest="tol",
    default="0.5",
    required=False,
    help="""The parameter to decide the flexibility while dealing the samples falling outside the bins.

""",
)
args = parser.parse_args()
datapath = args.data_dir
alpha = float(args.alpha)
tol = float(args.tol)
n_bins = args.n_bins
if n_bins != "auto":
    n_bins = int(n_bins)

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

classifiers = {
    "HBOS": HBOS(n_bins=n_bins, alpha=alpha, tol=tol, contamination=outlier_fraction)
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
    dump(clf, cnvrg_workdir + "/clf.joblib")
