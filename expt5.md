# EXPERIMENT 5 
## Introduction 
This repository contains the code and results for the foyrth experiment in the Data Science Fundamentals with Python course. The objective of this experiment is to data visualization using matplot library part-1 using data set obtained from UCIML repository 
## Steps-

### 1.Import Libraries:

Import necessary libraries: pandas for data handling, matplotlib for plotting, and seaborn for advanced visualizations.

### 2.Load Dataset:
Load the dataset using pd.read_csv(), specifying the correct delimiter (semicolon ; in this case).

### 3.Clean Column Names:
Strip any leading or trailing spaces from the column names to avoid reference issues when accessing columns.

### 4.Check Data:

Inspect the first few rows (head()) and column names (columns) to understand the dataset structure.

### 5.Handle Missing Values:

Fill any missing values in numeric columns with the mean of each respective column.
### 6.Plotting 
### a)Histogram:

Plot a histogram of the alcohol column to visualize its distribution.

### b)Box Plot:

Create a box plot for the fixed acidity column to identify its distribution and potential outliers.

### c)Scatter Plot:
Plot a scatter plot between fixed acidity and citric acid to see their relationship.

### d)Correlation Heatmap:

Generate a heatmap to show the correlation between different numeric features in the dataset.

### 7.Data Insights:

Check for missing values in the dataset and print the summary statistics for the numeric columns.


## Concepts used :

Data Loading: Load the dataset from a CSV file.

Data Cleaning: Clean column names and fill missing values with the mean.

Data Visualization:
Histograms: Show distribution of a feature.
Box Plots: Show spread and outliers.
Scatter Plots: Show relationships between two features.
Correlation Heatmap: Show correlations between features.

EDA: Check for missing values and generate summary statistics.


```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame from the provided path, specifying the delimiter (semicolon)
data = pd.read_csv('/content/drive/MyDrive/winequality-white.csv', sep=';')  # Adjust path if necessary

# Strip any leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# View the first few rows of the dataset (optional)
print(data.head())

# Print the column names to inspect their format
print(data.columns)

# Select only numeric columns, handling potential spaces in column names
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Fill missing values in numeric columns with the mean of each column
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# ========================
# Visualization Section
# ========================

# 1. Plot a histogram of the 'alcohol' column
if 'alcohol' in data.columns:
    plt.hist(data['alcohol'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Alcohol Content')
    plt.xlabel('Alcohol')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("'alcohol' column not found in the dataset.")

# 2. Box plot for 'fixed acidity'
# Access the column using its actual name from the printed columns
plt.figure(figsize=(6, 4))
plt.boxplot(data['fixed acidity'], vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
plt.title('Box plot of Fixed Acidity')
plt.xlabel('Fixed Acidity')
plt.show()

# 3. Scatter plot between 'fixed acidity' and 'citric acid'
# Access columns using their actual names
plt.figure(figsize=(6, 4))
plt.scatter(data['fixed acidity'], data['citric acid'], color='purple', alpha=0.5)
plt.title('Scatter Plot of Fixed Acidity vs Citric Acid')
plt.xlabel('Fixed Acidity')
plt.ylabel('Citric Acid')
plt.show()

# 4. Correlation Heatmap to visualize relationships between features
corr_matrix = data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# ========================
# More analysis and insights
# ========================

# Check for missing values
print("Missing values in dataset:")
print(data.isnull().sum())

# Summary of the dataset
print("Summary of the dataset:")
print(data.describe())



```

