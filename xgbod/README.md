# XGBOD
XGBOD class for outlier detection. It first uses the passed in unsupervised outlier detectors to extract richer representation of the data and then concatenates the newly generated features to the original feature for constructing the augmented feature space. An XGBoost classifier is then applied on this augmented feature space. 
XGBOD is the only supervised algorithm in our set of selected anomaly detection algorithms. Supervised algorithms outperform unsupervised algorithms when we are detecting outliers/anomalies that are similar to previously seen outliers/anomalies.

### Inputs

  - `--data_dir` : represents the path to the csv file taken as input from the output artifacts of preprocess library.
  - `--n_estimators` : Number of boosted trees to fit.
  - `--max_depth` : Maximum tree depth for base learners.
  - `--learning_rate` : Boosting learning rate 


### Outputs
The output model will be saved inside as **xgbod.joblib** file. This can be reused in the future to make predictions.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/xgbod/model.py --data_dir {path to input csv} --learning_rate {learning rate} --max_depth {tree depth for base learners} --n_estimators {number of boosted trees}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/xgbod/model.py --data_dir /input/preprocess/data_df_pca.csv --learning_rate 0.1 --max_depth 7 --n_estimators 150
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod

