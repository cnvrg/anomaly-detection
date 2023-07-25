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

from pyod.models.auto_encoder import AutoEncoder
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    average_precision_score,
)
from cnvrg import Experiment
from keras.losses import binary_crossentropy
from keras.losses import mean_squared_error
from sklearn.metrics import roc_auc_score
import numpy as np
import os

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


##########hyperparams for optimization#########
parser.add_argument(
    "--hidden_neurons",
    action="store",
    dest="hidden_neurons",
    default="3",
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



args = parser.parse_args()
datapath = args.data_dir
epochs = int(args.epochs)
dropout_rate = float(args.dropout_rate)
l2_regularizer = float(args.l2_regularizer)
hidden_neurons = args.hidden_neurons
if args.hidden_neurons != "default":
    hidden_neurons = int(args.hidden_neurons)
    hidden_neurons = [x for x in range(5, 1, -1)]
    hidden_neurons.extend([0.5])
    hidden_neurons = np.array(hidden_neurons)

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
    # create symmetric neural network
    hidden_neurons = list(hidden_neurons * trainX.shape[1])
    hidden_neurons[-1] = max(trainX.shape[1] // 2, 1)
    hidden_neuronsfinal = hidden_neurons.copy()
    hidden_neurons.reverse()
    hidden_neuronsfinal.extend(hidden_neurons)
    classifiers = {
        "autoencoder": AutoEncoder(
            hidden_neurons=hidden_neuronsfinal,
            hidden_activation="relu",
            output_activation="sigmoid",
            optimizer="adam",
            epochs=epochs,
            batch_size=32,
            dropout_rate=dropout_rate,
            l2_regularizer=l2_regularizer,
            validation_size=0.1,
            preprocessing=False,
            verbose=1,
            random_state=None,
            contamination=outlier_fraction,
        )
    }
else:
    classifiers = {
        "autoencoder": AutoEncoder(
            hidden_neurons=[
                64,
                32,
                max(trainX.shape[1] // 2, 1),
                max(trainX.shape[1] // 2, 1),
                32,
                64,
            ],
            hidden_activation="relu",
            output_activation="sigmoid",
            optimizer="adam",
            epochs=epochs,
            batch_size=32,
            dropout_rate=dropout_rate,
            l2_regularizer=l2_regularizer,
            validation_size=0.1,
            preprocessing=False,
            verbose=1,
            random_state=None,
            contamination=outlier_fraction,
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
    clf.model_.save(cnvrg_workdir + "/clf")
