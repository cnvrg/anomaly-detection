Use this training blueprint to train multiple algorithms and identify the best performer to detect data anomalies or outliers. This blueprint also establishes an endpoint that can be used to make inferences for anomaly detections based on the newly trained model.

This is a one-click, train-and-deploy blueprint for predicting data anomalies. Place the labeled input data in CSV format in a connector such as S3 or Splunk, which the blueprint uses to train multiple algorithms. The best performing algorithms are selected and deployed as an API endpoint. Then, users can call this API on new data points and view responses whether the new data points are outliers.

Complete the following steps to train an anomaly-detector model:

1. Click the **Use Blueprint** button.
2. In the dialog, select the relevant compute to deploy the API endpoint and click the **Start** button.
3. The cnvrg software redirects to your endpoint. Complete one or both of the following options:
   - Use the Try it Live section with any input data point to check your model.
   - Use the bottom integration panel to integrate your API with your code by copying in the code snippet.

An API endpoint that detects whether an input data point is an anomaly has now been deployed. Click [here]() for detailed instructions to run this blueprint. To learn how this blueprint was created, click [here](https://github.com/cnvrg/anomaly-detection).
