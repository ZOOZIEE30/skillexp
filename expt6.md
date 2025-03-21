# EXPERIMENT 7 
## Introduction 
This repository contains the code and results for the Feature Selection experiment in the Data Science Fundamentals with Python course. The objective of this experiment is to apply various feature selection techniques to identify the most relevant features for building machine learning models. This experiment uses a dataset obtained from the UCI Machine Learning Repository.
## Steps-

### 1.Import Libraries:

Import the necessary libraries: pandas for data handling, seaborn and matplotlib for visualizations, and machine learning libraries like sklearn for applying feature selection methods.

### 2.Load Dataset:
Load the dataset using pd.read_csv(), specifying the correct delimiter (semicolon ; in this case).

### 3.Clean Column Names:
Strip any leading or trailing spaces from the column names to avoid reference issues when accessing columns.

### 4.Check Data:

Inspect the first few rows (head()) and column names (columns) to understand the dataset structure.

### 5.Handle Missing Values:

Fill any missing values in numeric columns with the mean of each respective column.
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset (replace with actual dataset path if needed)
df = pd.read_csv('/content/drive/MyDrive/iris.data', header=None)

# Assign column names based on the Iris dataset
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Clean column names (remove spaces, convert to lowercase)
df.columns = df.columns.str.strip().str.lower()

# Encode the target variable 'species' to numeric values
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Check the first few rows of the dataset
print(df.head())

# Load the Iris dataset (replace with actual dataset path if needed)
df = pd.read_csv('/content/drive/MyDrive/iris.data', header=None)

# Assign column names based on the Iris dataset
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Clean column names (remove spaces, convert to lowercase)
df.columns = df.columns.str.strip().str.lower()

# Encode the target variable 'species' to numeric values
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Check the first few rows of the dataset
print(df.head())
```
### 6. Feature Selection Techniques 
### a) Filter Method

We can use a correlation matrix to check which features are most related to each other and to the target variable.
```
# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```
### b) Wrapper Method: Recursive Feature Elimination (RFE
Apply RFE with a LogisticRegression model to select the top 3 features.
```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Separate the features and target variable
X = df.drop('species', axis=1)
y = df['species']

# Create a logistic regression model
model = LogisticRegression(max_iter=200)

# Apply RFE to select the top 3 features
selector = RFE(model, 3)
selector = selector.fit(X, y)

# Print the features selected by RFE
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features)
```
### c) Embedded Method: Feature Importance Using Random Forest
Use Random Forest to compute feature importance, which is an embedded method of feature selection.
```
from sklearn.ensemble import RandomForestClassifier

# Train a random forest classifier
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Print the feature importance values
print("Feature Importances:")
print(feature_importance)
```

## Concepts used :


Data Loading: Loaded the dataset using pandas.read_csv().
Data Cleaning: Cleaned column names and handled missing values if necessary.
Feature Selection:
Filter Method: Correlation matrix to assess relationships between features.
Wrapper Method: Recursive Feature Elimination (RFE) to iteratively eliminate least important features.
Embedded Method: Random Forest for feature importance to assess which features are most useful for classification.
