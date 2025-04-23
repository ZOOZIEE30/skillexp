## Introduction 
This repository contains the code and results for the fourth experiment in the *Data Science Fundamentals with Python* course. The objective of this experiment is to learn **data visualization using the Matplotlib library (Part-1)**, using a dataset obtained from the UCIML repository.

## Steps-

### 1. Import Libraries:
Import necessary libraries:
- `pandas` for data handling and manipulation.
- `matplotlib` for basic plotting.
- `seaborn` for enhanced visualizations with less code.

These tools are standard in Python data analysis and allow for flexible and powerful visual exploration of datasets.

### 2. Load Dataset:
We use `pd.read_csv()` to load the dataset, making sure to set the correct delimiter as `;` (semicolon), because the data is formatted that way.

### 3. Clean Column Names:
Often, column names may have trailing spaces which can cause issues when accessing them. We use `.str.strip()` to clean them.

### 4. Check Data:
Use `.head()` to preview the top rows and `.columns` to list all column names. This helps us understand the structure and content of the dataset before analysis.

### 5. Handle Missing Values:
Missing data can mislead analysis. We fill missing values in numeric columns using the **mean** of each column. This is a common and simple imputation method that preserves overall trends.

### 6. Plotting 

#### a) Histogram:
We plot a histogram of the `alcohol` column to observe the distribution. Histograms are useful for identifying skewness, modality (uni/bi-modal), and general spread of data.

#### b) Box Plot:
A box plot is used for `fixed acidity` to check spread and detect potential outliers. The box shows the interquartile range and the line inside represents the median.

#### c) Scatter Plot:
We use a scatter plot between `fixed acidity` and `citric acid` to understand if there's a correlation between them. Scatter plots are ideal for spotting relationships between two continuous variables.

#### d) Correlation Heatmap:
This heatmap visualizes how strongly each numeric column is related to the others using color gradients. It helps in identifying strong positive or negative relationships among features.

### 7. Data Insights:
We print out:
- Missing values using `.isnull().sum()` to check for any remaining gaps in the dataset.
- Summary statistics using `.describe()` to get the count, mean, standard deviation, and range for each numeric column.

## Concepts Used:

**Data Loading**: Loading structured data from a CSV file into a pandas DataFrame.

**Data Cleaning**:
- Remove unwanted spaces from column headers.
- Fill missing values using column means for simplicity and consistency.

**Data Visualization**:
- **Histograms** show the frequency distribution of a single feature.
- **Box Plots** illustrate the spread and detect outliers visually.
- **Scatter Plots** reveal relationships between two numeric features.
- **Correlation Heatmaps** give a matrix-like view of feature-to-feature correlation.

**Exploratory Data Analysis (EDA)**:
- Checking for null values.
- Viewing summary statistics to understand data spread and central tendency.

## Code:

