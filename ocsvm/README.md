# OCSVM
A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier. This is used for anomaly detection.
### Inputs

  - `--data_dir` : represents the path to the csv file taken as input from the output artifacts of preprocess library.
  - `--tol` : Tolerance for stopping criterion.
  - `--kernel` : Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
  - `--nu` : An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1].


### Outputs
The output model will be saved inside as **clf.joblib** file. This can be reused in the future to make predictions.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/ocsvm/model.py --data_dir {path to input csv} --alpha {aplha value} --data_dir {path to input file} ---kernel {specify kernel type} --nu {upperbound on the fraction of train error} --tol {tolerance for stopping criteria}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/ocsvm/model.py --data_dir /input/preprocess/data_df_pca.csv --kernel rbf --nu 0.7 --tol 0.005
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod

