# Inference
User can use this deployment to make API calls to the trained anomaly detection model to get inference whether the new data point is an anomaly or not. The endpoint takes features as input and returns 1 for anomaly and 0 for normal.
The endpoint uses best of the unsupervised models and best of the XGBOD supervised model to make the prediction. Prediction from both the algorithms is joined by logical OR operation. If any of the two algorithms declares the new datapoint as an anomaly, the final prediction from the endpoint will be 1 (meaning the datapoint is an anomaly.)
We are using both supervised and unsupervised algorithms for prediction and combining their results.
The unsupervised algorithms are better at detecting new kinds of anomalies and supervised algorithms are better at detecting anomalies that resemble the ones that have already occured in the past.

## Input Arguments
```
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"vars": [0.0, -1.3598071336738, "yes", "male",-0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]}'

```

In case any of the feature values are missing for the data point for which you want prediction, you can pass empty "" value in place of that feature. For example in the following example value for third feature is not available:
```
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"vars": [0.0, -1.3598071336738, "", "male",-0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]}'

```


## Output
 
 ```
 {"prediction":{"anomaly":"1"}}
 ```