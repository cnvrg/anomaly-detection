This blueprint allows is a one click train and deploy solution for making Anomaly prediction. The user needs to provide the labelled input data in the .csv format which is used to train multiple algorithms. The best performing algorithms are selected and deployed as an API. User can proceed to call this API on new data points and get a response on whether the new data point is an anomaly or not.


1. Click on `Use Blueprint` button.
2. In the pop up, choose the relevant compute you want to use to deploy your API endpoint.
3. You will be redirected to your endpoint
4. You can now use the `Try it Live` section with any image.
5. You can now integrate your API with your code using the integration panel at the bottom of the page.
6. You will now have a functioning API endpoint that returns whether the input point is an anomaly or not.
   
Please note: This blueprint involves running multiple experiments to determine the best algorithm and best version of the that algorithm to use for API deployment based on your dataset. Running this blueprint might exhaust all your free resources if you are using the community version.

[See more about this blueprint here.](https://github.com/cnvrg/anomaly-detection)

