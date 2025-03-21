# EXPERIMENT 8: Feature Selection Techniques (Part 2)
## Introduction
 
This repository contains the code and results for Experiment 7 in the Data Science Fundamentals with Python course. The objective of this experiment is to apply advanced feature selection techniques to the Iris Dataset from the UCI Machine Learning Repository. We will use several techniques such as Univariate Feature Selection (SelectKBest), Recursive Feature Elimination (RFE), and Feature Importance using Random Forest to identify the most relevant features in the dataset.
## Steps-

### 1.Import Libraries:

Import necessary libraries: pandas for data handling, matplotlib for plotting, and seaborn for advanced visualizations.
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
```

### 2.Load Dataset:
Load the dataset using pd.read_csv(), specifying the correct delimiter (semicolon ; in this case).
```
# Load Iris dataset from UCI Machine Learning Repository
df = pd.read_csv('/content/drive/MyDrive/iris.data', header=None)
```

### 3. Data Preprocessing:
We separate the dataset into features (X) and target variable (y) and then split it into training and test datasets. We also standardize the feature data.

```
# Features (X) and Target (y)
X = df.drop('species', axis=1)
y = df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4.Univariate Feature Selection (SelectKBest)::
In this step, we use SelectKBest with the ANOVA F-test (f_classif) to select the top 2 features from the dataset.
```
# Apply SelectKBest with ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)

# Show the selected features
selected_features = pd.DataFrame(selector.scores_, columns=['Score'], index=X.columns)
print("Selected Features based on Univariate Selection:")
print(selected_features.nlargest(2, 'Score'))  # Show top 2 features
```
### 5. Recursive Feature Elimination (RFE):
In this step, we use Recursive Feature Elimination (RFE) with Logistic Regression as the estimator to recursively remove the least important features and select the top 2.
```
# Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe = RFE(model, 2)  # Select top 2 features
X_train_rfe = rfe.fit_transform(X_train, y_train)

# Show which features are selected
print("Selected Features based on RFE:")
selected_rfe_features = pd.DataFrame({'Feature': X.columns, 'Rank': rfe.ranking_})
print(selected_rfe_features[selected_rfe_features['Rank'] == 1])  # Features with rank 1 are selected
```

### 6. Feature Importance using Random Forest:
Here, we use a Random Forest classifier to calculate the feature importances. This method uses tree-based models to assess how important each feature is for making predictions.
```
# Feature importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print("Feature Importances based on Random Forest:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values(by='Importance', ascending=False))
plt.title("Feature Importances based on Random Forest")
plt.show()
```

### 7. Model Evaluation Using Selected Features:
In this step, we evaluate the performance of a Random Forest classifier using the top 2 selected features from Univariate Selection.

```
# Using the top 2 selected features based on Univariate Selection
X_train_selected_features = X_train[:, selector.get_support()]  # Selecting the columns with True values
X_test_selected_features = X_test[:, selector.get_support()]

# Train a Random Forest classifier
rf.fit(X_train_selected_features, y_train)

# Predict on the test set
y_pred = rf.predict(X_test_selected_features)

# Evaluate the model
print("Accuracy on Test Data with Selected Features (Univariate Selection):")
print(accuracy_score(y_test, y_pred))
```



## Concepts used :

In Part 2 of Feature Selection, we applied several advanced feature selection techniques:

Univariate Feature Selection (SelectKBest) – This method evaluates features using statistical tests like the ANOVA F-value and selects the most relevant ones.
Recursive Feature Elimination (RFE) – This technique recursively eliminates the least important features, based on an estimator like Logistic Regression, to find the optimal set of features.
Feature Importance from Random Forest – This method calculates the importance of each feature using a tree-based model like Random Forest, and ranks them accordingly.
These techniques allow us to identify the most relevant features in the dataset, which can help improve the performance of machine learning models by reducing overfitting, increasing efficiency, and enhancing generalization.

