# Iforest
The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. This path length, averaged over a forest of such random trees, is a measure of normality and our decision function. Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies. 

### Inputs

  - `--data_dir` : represents the path to the csv file taken as input from the output artifacts of preprocess library.
  - `--n_estimators` : The number of base estimators in the ensemble.


### Outputs
The output model will be saved inside as **clf.joblib** file. This can be reused in the future to make predictions.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/Iforest/model.py --data_dir {path to input csv} --alpha {aplha value} --data_dir {path to input file} --n_estimators {number of estimators}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/Iforest/model.py --data_dir /input/preprocess/data_df_pca.csv --n_estimators 150
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod

