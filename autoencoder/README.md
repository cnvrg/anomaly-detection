# AutoEncoder
Autoencoders are used to learn the representation of data in unsupervised manner. They can be used to detect the outliers in the dataset by using reconstruction errors. Autoencoders take the features as input without the labels. Along with this, we need to input the contamination in the dataset which is used to fit a threshold to classifying data points as outliers. 
Autoendcoders are symmetric neural networks. They take the features as input and project them onto lower dimensions, then try to reconstruct the features back to the original dimension space. While learning the reconstruction between input and output features is minimised. Since the number of outliers/anomaly points are much lesser than normal points, the neural network will not learn to reconstruct these points efficiently

### Inputs

  - `--data_dir` : represents the path to the csv file taken as input from the output artifacts of preprocess library.
  - `--hidden_neurons` : Integer input describes the layer structure of the neural network. For example if the input is 3, then the architecture of the neural network will be: **L = total_number_of_data_features** 
  3*[L],2*[data_features],0.5*[L],0.5*[L],2*[L],3*[L]
if the input is 5, then the architecture of the neural network will be: 
5*[L],4*[L],3*[L],2*[L],0.5*[L],0.5*[L],2*[L],3*[L],4*[L],5*[L]
  - `epochs** : Total number of passes over the training data while training.
  - `--dropout_rate` : The dropout to be used across all layers.
  - `--l2_regularizer` : The l2 regularization rate to be used.
   
### Outputs
The output model will be saved inside the **clf** directory. This can be reused in the future to make predictions.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/autoencoder/model.py --data_dir {path to input csv} --dropout_rate {dropout rate} --epochs {number of epochs} --hidden_neurons {hidden neuron size} --l2_regularizer {regularization rate}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.Large" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/autoencoder/model.py --data_dir /input/preprocess/data_df_pca.csv --dropout_rate 0.4 --epochs 300 --hidden_neurons default --l2_regularizer 0.3
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod

