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
