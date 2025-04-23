
## Introduction
This repository contains the code and results for an experiment involving **data visualization and statistical analysis** using Python. The objective is to explore a dataset by performing summary statistics and visualizations such as histograms, box plots, and correlation heatmaps. These techniques are core parts of Exploratory Data Analysis (EDA) in data science.

## Steps-

### 1. Import Libraries:
We import:
- `pandas` for handling datasets,
- `matplotlib.pyplot` and `seaborn` for creating plots,
- `numpy` for numerical operations,
- and `warnings` to ignore unnecessary warnings during execution.

### 2. Load Dataset:
The dataset is loaded using `pd.read_csv()` and stored in a DataFrame for manipulation.

### 3. Data Overview:
We use `.head()` and `.info()` to understand the dataset's structure, types, and non-null counts.

### 4. Data Cleaning:
- Unnecessary columns are dropped.
- Missing values are identified.
- Nulls in important features are filled using mean or mode values.

### 5. Visualization:

#### a) Histograms:
Histograms are plotted to understand the distribution of numerical columns.

#### b) Box Plots:
Box plots help us identify outliers and the spread of key numeric variables.

#### c) Count Plots:
These plots are used to visualize categorical distributions.

#### d) Correlation Heatmap:
This shows the correlation between features, helping identify relationships.

### 6. Data Insights:
- Null values are checked and handled.
- Summary statistics are generated using `.describe()`.

## Concepts Used:

- **Data Loading**: Reading CSV file into a pandas DataFrame.
- **Missing Data Handling**: Using `.isnull().sum()` and `.fillna()` techniques.
- **Data Visualization**:
  - Histogram for distributions
  - Boxplot for spread and outliers
  - Countplot for categorical counts
  - Heatmap for feature correlation
- **Statistical Summary**: Descriptive stats using `.describe()`.

## Code & Output:

```python
# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("/content/LoanApprovalPrediction.csv")
df.head()
```

```output
   Loan_ID Gender Married Dependents Education Self_Employed  ApplicantIncome  ...
0  LP001002   Male     No          0  Graduate            No             5849
1  LP001003   Male    Yes          1  Graduate            No             4583
...
```

```python
# Checking data structure
df.info()
```

```output
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 614 entries, 0 to 613
Data columns (total 13 columns):
...
```

```python
# Count of missing values
df.isnull().sum()
```

```output
Gender               13
Married               3
Dependents           15
...
```

```python
# Filling missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
...
```

```python
# Histogram
df['ApplicantIncome'].hist(bins=50)
plt.title("Applicant Income Distribution")
plt.show()
```

```python
# Box Plot
sns.boxplot(x='Education', y='ApplicantIncome', data=df)
plt.title("Income vs Education")
plt.show()
```

```python
# Count Plot
sns.countplot(x='Loan_Status', hue='Gender', data=df)
plt.title("Loan Approval by Gender")
plt.show()
```

```python
# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

```python
# Summary Statistics
df.describe()
```

```output
       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History
count       614.000000         614.000000  614.000000         614.000000      614.000000
mean        5403.459283        1621.245798  146.412162         342.000000        0.842199
...
```
