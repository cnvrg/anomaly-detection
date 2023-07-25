# Data Preprocessing
This library is used to preprocess the input csv to convert to into a suitable format for the machine learning algorithms.
📝 **Note**
 - The data provided should contain 70% - 95% normal examples and only 30% - 5% anomaly points.
 - The column in the csv denoting whether the point is an anomaly or not should be called **"Class"** and should be the last column in the csv. 1 means the point is anomaly and 0 means the point is normal.
 - You will need to provide a list of columns that need to label encoded, hot encoded, name of ID column if any.
 
### Features
- print a statistical summary of the data
- perform missing value treatment for each type of column in the data	
- define the mapping function to map yes and no's to 1's and 0's wherever required	

| Missing Value Treatment Logic |
| -------------------------------------- |
| Blank/Single Valued Columns :- Removed |
| Numerical Columns + Unique Value Count > 20% :- Mean |
| Numerical Columns + Unique Value Count < 20% :- Median |
| Numerica Columns + 1/0 Values :- Randomly filled with 1 or 0 |
| String Columns + Yes/No Values :- Randomly filled with Yes or No |
| Other String Columns :- Randomly filled with value "Garbage Value 999" |

- See if the user has defined minmax or standard scaler to be used	
- Perform encoding of character values 	

| Encoding Logic |
| -------------------------------------- |
| If column type ~ string and not in label_encoding_cols :- One Hot Encoding |
| If column type ~ string and column values are "Yes" and "No" then :- 1,0 Mapping |
| Other String columns :- Label encoding |

- Mark missing values of the dataframe as grabage values and list the indexes where these values lie 
- If id column and label encoding columns are labelled as "None" then they need to be done in all libraries where they are called from.
- For one hot encoding columns/ ordinal encoding columns perform the encoding on the train dataset and dump the encoders in the artifacts to be reused for the user's dataset 
- Print the processed data file, a dataframe containing the list of id & label encoding columns/standard scaler object	
- Do principal component analysis if the total number of features excluding ID column is more than 10. 

# Input Arguments
- `--anomaly_data` (/data/churn_data/churn.csv) -- raw data uploaded by the user on the platform
- `--id_column`(default None) -- get the name of the id column by the user
- `--label_encoding_cols`(default None) -- get list of columns to be label encoded by the user
- `--scaler`(default Minmax) -- set the type of scaler to be used
- `--pca_arg` value ranging from (0,1) is provided that is used to the set the amount of information captured after doing PCA. Default value is 0.95 which means 95% of the information from the original data is captured after PCA.
- `pca_overwrite` used to overwrite the decision whether to do PCA or not. By default overwrite is set to False. User can set it to True if you want to overwrite the decision taken whether to do PCA or not. We do PCA if the total number of features excluding ID column is more than 10. In case you want reverse of this(i.e you want PCA done but number of features is less than 10 or if you don't want PCA done and number of features is more than 10), you can set this param to True.

# Model Artifacts

- `--original_col.csv`	original file column names and average/random values as a single row

| customerID  | gender  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | MultipleLines  | InternetService  | OnlineSecurity  | OnlineBackup        | DeviceProtection  | TechSupport  | StreamingTV  | StreamingMovies  | Contract       | PaperlessBilling  | PaymentMethod           | MonthlyCharges    | TotalCharges       | Class  |
|-------------|---------|----------------|----------|-------------|---------|---------------|----------------|------------------|-----------------|---------------------|-------------------|--------------|--------------|------------------|----------------|-------------------|-------------------------|-------------------|--------------------|--------|
| 1867-TJHTS  | Female  | 0              | Yes      | Yes         | 29      | Yes           | No             | Fiber optic      | No              | No internet service | No                | No           | No           | Yes              | Month-to-month | Yes               | Credit card (automatic) | 64.76169246059918 | 2283.3004408418656 | Yes    |

- `--data_df.csv`	processed file (after one hot encoding, label encoding, missing value treatment)

| customerID  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | InternetService  | PaperlessBilling  | PaymentMethod  | MonthlyCharges      | TotalCharges         | Churn  | Contract-Month-to-month  | Contract-One year  | Contract-Two year  | DeviceProtection-No  | DeviceProtection-No internet service  | DeviceProtection-Yes  | MultipleLines-No  | MultipleLines-No phone service  | MultipleLines-Yes  | OnlineBackup-No  | OnlineBackup-No internet service  | OnlineBackup-Yes  | OnlineSecurity-No  | OnlineSecurity-No internet service  | OnlineSecurity-Yes  | StreamingMovies-No  | StreamingMovies-No internet service  | StreamingMovies-Yes  | StreamingTV-No  | StreamingTV-No internet service  | StreamingTV-Yes  | TechSupport-No  | TechSupport-No internet service  | TechSupport-Yes  | gender-Female  | gender-Male  |
|-------------|----------------|----------|-------------|---------|---------------|------------------|-------------------|----------------|---------------------|----------------------|--------|--------------------------|--------------------|--------------------|----------------------|---------------------------------------|-----------------------|-------------------|---------------------------------|--------------------|------------------|-----------------------------------|-------------------|--------------------|-------------------------------------|---------------------|---------------------|--------------------------------------|----------------------|-----------------|----------------------------------|------------------|-----------------|----------------------------------|------------------|----------------|--------------|
| 1867-TJHTS  | 0.0            | 1        | 1           | 1.0     | 1             | 1.0              | 1                 | 1.0            | 0.11542288557213931 | 0.001275098084468036 | 1      | 1.0                      | 0.0                | 0.0                | 1.0                  | 0.0                                   | 0.0                   | 1.0               | 0.0                             | 0.0                | 0.0              | 1.0                               | 0.0               | 1.0                | 0.0                                 | 0.0                 | 0.0                 | 0.0                                  | 1.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 1.0            | 0.0          |
| 5575-GNVDE  | 0.0            | 0        | 0           | 34.0    | 1             | 0.0              | 0                 | 3.0            | 0.3850746268656716  | 0.21586660512347106  | 0      | 0.0                      | 1.0                | 0.0                | 0.0                  | 0.0                                   | 1.0                   | 1.0               | 0.0                             | 0.0                | 1.0              | 0.0                               | 0.0               | 0.0                | 0.0                                 | 1.0                 | 1.0                 | 0.0                                  | 0.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 0.0            | 1.0          |
- `--processed_col.csv`	processed file (after one hot encoding, label encoding, missing value treatment) but with only 1 row (for batch predict)

| customerID  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | InternetService  | PaperlessBilling  | PaymentMethod  | MonthlyCharges      | TotalCharges         | Churn  | Contract-Month-to-month  | Contract-One year  | Contract-Two year  | DeviceProtection-No  | DeviceProtection-No internet service  | DeviceProtection-Yes  | MultipleLines-No  | MultipleLines-No phone service  | MultipleLines-Yes  | OnlineBackup-No  | OnlineBackup-No internet service  | OnlineBackup-Yes  | OnlineSecurity-No  | OnlineSecurity-No internet service  | OnlineSecurity-Yes  | StreamingMovies-No  | StreamingMovies-No internet service  | StreamingMovies-Yes  | StreamingTV-No  | StreamingTV-No internet service  | StreamingTV-Yes  | TechSupport-No  | TechSupport-No internet service  | TechSupport-Yes  | gender-Female  | gender-Male  |
|-------------|----------------|----------|-------------|---------|---------------|------------------|-------------------|----------------|---------------------|----------------------|--------|--------------------------|--------------------|--------------------|----------------------|---------------------------------------|-----------------------|-------------------|---------------------------------|--------------------|------------------|-----------------------------------|-------------------|--------------------|-------------------------------------|---------------------|---------------------|--------------------------------------|----------------------|-----------------|----------------------------------|------------------|-----------------|----------------------------------|------------------|----------------|--------------|
| 1867-TJHTS  | 0.0            | 1        | 1           | 1.0     | 1             | 1.0              | 1                 | 1.0            | 0.11542288557213931 | 0.001275098084468036 | 1      | 1.0                      | 0.0                | 0.0                | 1.0                  | 0.0                                   | 0.0                   | 1.0               | 0.0                             | 0.0                | 0.0              | 1.0                               | 0.0               | 1.0                | 0.0                                 | 0.0                 | 0.0                 | 0.0                                  | 1.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 1.0            | 0.0          |
- `--data_df_pca.csv`	processed file after pca. If PCA was not done this will be same as processed_col.csv . This is the final file after preprocessing that will be used by algorithms.

| customerID  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | InternetService  | PaperlessBilling  | PaymentMethod  | MonthlyCharges      | TotalCharges         | Churn  | Contract-Month-to-month  | Contract-One year  | Contract-Two year  | DeviceProtection-No  | DeviceProtection-No internet service  | DeviceProtection-Yes  | MultipleLines-No  | MultipleLines-No phone service  | MultipleLines-Yes  | OnlineBackup-No  | OnlineBackup-No internet service  | OnlineBackup-Yes  | OnlineSecurity-No  | OnlineSecurity-No internet service  | OnlineSecurity-Yes  | StreamingMovies-No  | StreamingMovies-No internet service  | StreamingMovies-Yes  | StreamingTV-No  | StreamingTV-No internet service  | StreamingTV-Yes  | TechSupport-No  | TechSupport-No internet service  | TechSupport-Yes  | gender-Female  | gender-Male  |
|-------------|----------------|----------|-------------|---------|---------------|------------------|-------------------|----------------|---------------------|----------------------|--------|--------------------------|--------------------|--------------------|----------------------|---------------------------------------|-----------------------|-------------------|---------------------------------|--------------------|------------------|-----------------------------------|-------------------|--------------------|-------------------------------------|---------------------|---------------------|--------------------------------------|----------------------|-----------------|----------------------------------|------------------|-----------------|----------------------------------|------------------|----------------|--------------|
| 1867-TJHTS  | 0.0            | 1        | 1           | 1.0     | 1             | 1.0              | 1                 | 1.0            | 0.11542288557213931 | 0.001275098084468036 | 1      | 1.0                      | 0.0                | 0.0                | 1.0                  | 0.0                                   | 0.0                   | 1.0               | 0.0                             | 0.0                | 0.0              | 1.0                               | 0.0               | 1.0                | 0.0                                 | 0.0                 | 0.0                 | 0.0                                  | 1.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 1.0            | 0.0          |
| 5575-GNVDE  | 0.0            | 0        | 0           | 34.0    | 1             | 0.0              | 0                 | 3.0            | 0.3850746268656716  | 0.21586660512347106  | 0      | 0.0                      | 1.0                | 0.0                | 0.0                  | 0.0                                   | 1.0                   | 1.0               | 0.0                             | 0.0                | 1.0              | 0.0                               | 0.0               | 0.0                | 0.0                                 | 1.0                 | 1.0                 | 0.0                                  | 0.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 0.0            | 1.0          |

- `--ordinal_enc`	label/ordinal encoder saved file after fitting the encoder on training data
- `--encoded_values_file`	one hot encoder saved file after fitting the encoder on training data
- `--columns_list.csv`	table of 3 columns, one hot encoded columns, label encoded columns and id columns

|    OHE_columns   | id_columns | label_encoded_columns |
|:----------------:|:----------:|:---------------------:|
| Contract         | customerID | InternetService       |
| DeviceProtection |            | PaymentMethod         |

- `--mis_col_type.csv`	columns categorised on what kind of missing value treatment they received, mean, median, random value?

| Mean  | 0-1           | Median  | Yes-No     | String     |
|-------|---------------|---------|------------|------------|
|       | SeniorCitizen |         | Partner    | customerID |
|       |               | tenure  | Dependents | gender     |
- `--std_scaler.bin`	standard scaler saved object after fitting scaler on training data
- `pca_model` if pca done, pca model is saved which will be used later for inference.
### How to run
```
cnvrg run  --datasets='[{id:{dataset name},commit:{commit id dataset}}]' --machine={compute size} --image={image to use} --sync_before=false python3 Train_Blueprint/data_preprocessing/data_preprocessing.py --anomaly_data {path to input data} --label_encoding {name of cols for label encoding} --scaler {type of scaler to be used} --id_column {id col} --dependent_column {dependent col name} --pca_arg {pca arg} --pca_overwrite {whether to overwrite pca}
```
Example run
```
cnvrg run  --datasets='[{id:"anomalydata",commit:"09d66f6e5482d9b0ba91815c350fd9af3770819b"}]' --machine="default.medium" --image=cnvrg:v5.0 --sync_before=false python3 Train_Blueprint/data_preprocessing/data_preprocessing.py --anomaly_data /data/anomalydata/anomaly.csv --label_encoding "PaymentMethod,InternetService" --scaler standard --id_column customerID --dependent_column Class --pca_arg 0.999 --pca_overwrite no
```
