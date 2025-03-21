# EXPERIMENT 4
<br> This repository contains the code and results for the foyrth experiment in the Data Science Fundamentals with Python course. The objective of this experiment is to perform data pre-processing (normalizing,scaling and balancing data set )on the csv file taken from UCIML 

## Steps-
<br>
Import Required Libraries:-
<br>
pandas: This library helps you handle and manipulate data easily.
MinMaxScaler and StandardScaler: These are from sklearn.preprocessing, and are used to scale the data to specific ranges.
SMOTE: This is used for balancing the dataset by generating synthetic data points for underrepresented classes.
train_test_split: This helps split your dataset into two parts: one for training the model and one for testing it.
<br>
Load the Data:
<br>
The code reads the dataset (winequality-red.csv) from the given file path using pandas.read_csv(). This converts the data into a DataFrame, which is like a table where each column represents a feature (like acidity, alcohol content, etc.), and the last column represents the target (wine quality).
<br>
Handle Missing Values:
<br>
If there are any missing values in the dataset, this code fills them with the average (mean) value of the respective column using data.fillna(). This ensures there are no gaps or missing data.
<br>
Normalize the Data:
<br>
Normalization means adjusting the data so that each feature is scaled to a range of 0 to 1.
The MinMaxScaler() is applied to the numeric features of the dataset (excluding the target column) to scale them. This helps in ensuring that all features contribute equally to the model and no feature has a larger impact due to its scale.
<br>
Scale the Data (Standardization):
<br>
Standardization means adjusting the data so that each feature has a mean of 0 and a standard deviation of 1.
The StandardScaler() is applied to the numeric features to scale them to a standard range. This is helpful for machine learning algorithms that assume normally distributed data.
<br>
Balance the Data with SMOTE:
<br>
Some datasets might have an imbalance, meaning some categories of data are overrepresented while others are underrepresented. In this case, the code uses SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic data points for the underrepresented class. This balances the dataset and helps the model perform better.
Split the Data into Training and Testing Sets:

The dataset is divided into two parts:
Training set: Used to train the model.
Test set: Used to evaluate the model's performance after training.
train_test_split() is used to split the data, with 80% of the data used for training and 20% for testing.
<br>
Output the Results:

The processed data (normalized, scaled, and balanced) is ready to be used for training a machine learning model.

<br>
concepts used :
<br>
Data Preprocessing: Cleaning and transforming raw data.
<br>
Pandas: Library for handling data in tables.
<br>
Normalization: Scaling values to [0, 1].
<br>
Standardization: Transforming data to have mean 0, std 1.
<br>
SMOTE: Balances datasets by creating synthetic data.
<br>
Train-Test Split: Dividing data into training and testing sets
<br>

<br>

```
from sklearn.preprocessing import MinMaxScaler
import numpy as np
data= np.array([[100],[200],[300],[400],[500]])
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data)
print(scaled_data)
```
<br>
