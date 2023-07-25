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

import pandas as pd
import os
import psutil
import time
from cnvrg import Experiment
import shutil
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

for k in os.environ.keys():
    if "PASSED_CONDITION" in k and os.environ[k] == "true":
        task_name = k.replace("CNVRG_", "").replace("_PASSED_CONDITION", "").lower()
        # copy files from the task_name
        if task_name == "autoencoder" or task_name == "vae" or task_name == "deepsvdd":
            shutil.move("/input/" + task_name + "/clf", cnvrg_workdir)
            threshold = os.environ["CNVRG_" + task_name.upper() + "_THRESHOLD"]
            df = pd.DataFrame({"winner": task_name, "threshold": threshold}, index=[0])
            df.to_csv("/cnvrg/winner_details.csv")
            # write the file with threshold and winner name
        else:
            shutil.move("/input/" + task_name + "/clf.joblib", cnvrg_workdir)
            df = pd.DataFrame({"winner": task_name}, index=[0])
            df.to_csv("/cnvrg/winner_details.csv")
            # write the file with winner name
        print("winner is: ", task_name)
