# Compare
Compare block is used to select the best performing version of an algorithm out of all versions of all the algorithms. This block is specifically used for all the unpsupervised algorithms. It selects one best performing unsupervised algorithm based on the metric **average precision score**. AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight. You can learn more about it [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

### Inputs

All unsupervised libraries are attached to this task.

### Outputs
The output of this task is the best performing model.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/compare/compare.py
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/compare/compare.py
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod


