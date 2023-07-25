# HBOS
Histogram- based outlier detection (HBOS) is an efficient unsupervised method. It assumes the feature independence and calculates the degree of outlyingness by building histograms. Two versions of HBOS are supported: - Static number of bins: uses a static number of bins for all features. - Automatic number of bins: every feature uses a number of bins deemed to be optimal according to the [Birge-Rozenblac method](http://www.numdam.org/item/10.1051/ps:2006001.pdf). 

### Inputs

  - `--data_dir` : represents the path to the csv file taken as input from the output artifacts of preprocess library.
  - `--n_bins` : The number of bins for each feature.
  - `--alpha` : The regularizer for preventing overflow.
  - `--tol` : The parameter to decide the flexibility while dealing the samples falling outside the bins.

   
### Outputs
The output model will be saved inside as **clf.joblib** file. This can be reused in the future to make predictions.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/HBOS/model.py --data_dir {path to input csv} --alpha {aplha value} --data_dir {path to input file} --n_bins {bin size} --tol {tolerance value for bins}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/HBOS/model.py --alpha 0.5 --data_dir /input/preprocess/data_df_pca.csv --n_bins 30 --tol 0.7
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod

