# DeepSVDD
Deep One-Class Classifier with AutoEncoder (AE) is a type of neural networks for learning useful data representations in an unsupervised way. DeepSVDD trains a neural network while minimizing the volume of a hypersphere that encloses the network representations of the data, forcing the network to extract the common factors of variation. DeepSVDD could be used to detect outlying objects in the data by calculating the distance from center for details. 

### Inputs

  - `--data_dir` : represents the path to the csv file taken as input from the output artifacts of preprocess library.
  - `--hidden_neurons` : Integer input describes the layer structure of the neural network. For example if the input is 3, then the architecture of the neural network will be: **L = total_number_of_data_features** 
  3*[L],2*[data_features],0.5*[L]
if the input is 5, then the architecture of the neural network will be: 
5*[L],4*[L],3*[L],2*[L],0.5*[L]
  - `--epochs` : Total number of passes over the training data while training.
  - `--dropout_rate` : The dropout to be used across all layers.
  - `--l2_regularizer` : The l2 regularization rate to be used.
  - `--use_ae` : Autoencoder style neural network. It creates a symmetric representation if set to True. By default set to false.
   
### Outputs
The output model will be saved inside the **clf** directory. This can be reused in the future to make predictions.

### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/deepSVDD/model.py --data_dir {path to input csv} --dropout_rate {dropout rate} --epochs {number of epochs} --hidden_neurons {hidden neuron size} --l2_regularizer {regularization rate}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/deepSVDD/model.py --data_dir /input/preprocess/data_df_pca.csv --dropout_rate 0.3 --epochs 300 --hidden_neurons  7 --l2_regularizer 0.3
```
### Reference 
Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
https://github.com/yzhao062/pyod

