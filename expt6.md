# EXPERIMENT 6: Data Visualization with Matplotlib - Part 2 
## Introduction 
This repository contains the code and results for Experiment 6 in the Data Science Fundamentals with Python course. In this experiment, we explore advanced data visualization techniques using the Matplotlib library with the Iris dataset obtained from the UCI Machine Learning Repository. We will perform some visualizations that include scatter plots, histograms, box plots, and pair plots to better understand the relationships between features and the distribution of the data.
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
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load Iris dataset from UCI Machine Learning Repository
df = pd.read_csv('/content/drive/MyDrive/iris.data', header=None)

# Assign column names based on the Iris dataset
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Display the first few rows of the dataset
print(df.head())
# Clean column names
df.columns = df.columns.str.strip().str.lower()
```
### 6.Visualizing Data 
### a) Pair Plot

A pair plot provides a grid of scatter plots to show the relationships between all pairs of features in the dataset. This will help us understand the relationships between sepal and petal lengths and widths for the different species.
```
# Pairplot to show relationships between features
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()
```

### b)Histogram:

A histogram is a great way to visualize the distribution of a single feature. Let's plot the distribution of the sepal length
```
# Histogram for Sepal Length
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal_length'], bins=20, kde=True, color='blue', edgecolor='black')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()
```

### c)Box Plot
A box plot helps to visualize the spread of data, the median, and any outliers. Here, we will visualize the petal length of the flowers for each species.
```
# Box plot for Petal Length by Species
plt.figure(figsize=(8, 5))
sns.boxplot(x="species", y="petal_length", data=df, palette="Set2")
plt.title("Box Plot of Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()
```

### d)Violin Plot

A violin plot is similar to a box plot but provides more information about the distribution of the data, including density. We will plot the petal width by species..
```
# Violin plot for Petal Width by Species
plt.figure(figsize=(8, 5))
sns.violinplot(x="species", y="petal_width", data=df, inner="quart", palette="Set2")
plt.title("Violin Plot of Petal Width by Species")
plt.xlabel("Species")
plt.ylabel("Petal Width (cm)")
plt.show()
```

### e) Correlation Heatmap

A correlation heatmap shows the correlation matrix of features and helps to visualize how strongly features are related to each other. We'll plot the correlation matrix for all features in the dataset.

```
# Compute the correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Iris Dataset Features")
plt.show()
```
### f)Scatter Plot
A scatter plot can be used to examine the relationship between two continuous variables. We'll visualize the relationship between sepal length and sepal width.

```
# Scatter plot of Sepal Length vs Sepal Width
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal_length', y='sepal_width', data=df, hue='species', palette="Set2")
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
```

## Concepts used :

Concepts Used:
Data Loading: Loaded the dataset from the UCI repository.
Data Cleaning: Cleaned column names and handled the data for visualization.
Data Visualization:
Pair Plots: Visualized relationships between all pairs of features.
Histograms: Visualized distributions of individual features.
Box Plots: Visualized spread, median, and outliers of features.
Violin Plots: Showed distribution of features by category.
Scatter Plots: Examined relationships between two features.
Correlation Heatmap: Visualized correlation between features.
