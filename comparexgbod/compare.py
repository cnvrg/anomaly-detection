import pandas as pd
import os
import psutil
import time
from cnvrg import Experiment
import shutil
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

for k in os.environ.keys():
    if 'PASSED_CONDITION' in k and os.environ[k] == 'true':
        task_name = k.replace('CNVRG_', '').replace(
            '_PASSED_CONDITION', '').lower()
        #copy files from the task_name
        shutil.move("/input/xgbod/xgbod.joblib",cnvrg_workdir)
#read model here