
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
###Libraries used and Merging and Loading the dataset  
```
import pandas as pd
import glob
import os

# Get all CSV files in the current directory
csv_files = glob.glob('/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_1.csv')
csv_files = glob.glob('/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_2.csv')
csv_files = glob.glob('/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_3.csv')
csv_files = glob.glob('/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_4.csv')

# Step 2: Parameters
chunk_size = 100000  # Adjust based on memory
data_chunks = []

print(" Starting to read and combine all datasets...")

# Step 3: Read each file in chunks and append to the list
for file in csv_files:
    print(f" Processing file: {file}")
    try:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            data_chunks.append(chunk)
    except Exception as e:
        print(f" Error reading {file}: {e}")

# Step 4: Concatenate all chunks (handles different columns automatically)
print("🔗 Concatenating all chunks into one DataFrame...")
combined_df = pd.concat(data_chunks, ignore_index=True, sort=True)

# Step 5: Save to a new CSV
output_file = 'combined_dataset.csv'
combined_df.to_csv(output_file, index=False)
print(f" Done! Combined dataset with all rows and attributes saved as '{output_file}'")

# Step 6: Sanity check
print("\n Final dataset info:")
print(f"Total rows: {combined_df.shape[0]}")
print(f"Total columns: {combined_df.shape[1]}")
print(f"Column names: {combined_df.columns.tolist()}")
```
#### output 
```
Starting to read and combine all datasets...
 Processing file: /content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_4.csv
<ipython-input-3-7200f5717dd2>:21: DtypeWarning: Columns (7,9) have mixed types. Specify dtype option on import or set low_memory=False.
  for chunk in pd.read_csv(file, chunksize=chunk_size):
<ipython-input-3-7200f5717dd2>:21: DtypeWarning: Columns (7,9) have mixed types. Specify dtype option on import or set low_memory=False.
  for chunk in pd.read_csv(file, chunksize=chunk_size):
<ipython-input-3-7200f5717dd2>:21: DtypeWarning: Columns (7,9) have mixed types. Specify dtype option on import or set low_memory=False.
  for chunk in pd.read_csv(file, chunksize=chunk_size):
<ipython-input-3-7200f5717dd2>:21: DtypeWarning: Columns (7,9) have mixed types. Specify dtype option on import or set low_memory=False.
  for chunk in pd.read_csv(file, chunksize=chunk_size):
<ipython-input-3-7200f5717dd2>:21: DtypeWarning: Columns (7,9) have mixed types. Specify dtype option on import or set low_memory=False.
  for chunk in pd.read_csv(file, chunksize=chunk_size):
🔗 Concatenating all chunks into one DataFrame...
 Done! Combined dataset with all rows and attributes saved as 'combined_dataset.csv'

 Final dataset info:
Total rows: 668522
Total columns: 46
Column names: ['AR_P_Proto_P_Dport', 'AR_P_Proto_P_DstIP', 'AR_P_Proto_P_Sport', 'AR_P_Proto_P_SrcIP', 'N_IN_Conn_P_DstIP', 'N_IN_Conn_P_SrcIP', 'Pkts_P_State_P_Protocol_P_DestIP', 'Pkts_P_State_P_Protocol_P_SrcIP', 'TnBPDstIP', 'TnBPSrcIP', 'TnP_PDstIP', 'TnP_PSrcIP', 'TnP_PerProto', 'TnP_Per_Dport', 'attack', 'bytes', 'category', 'daddr', 'dbytes', 'dpkts', 'dport', 'drate', 'dur', 'flgs', 'flgs_number', 'ltime', 'max', 'mean', 'min', 'pkSeqID', 'pkts', 'proto', 'proto_number', 'rate', 'saddr', 'sbytes', 'seq', 'spkts', 'sport', 'srate', 'state', 'state_number', 'stddev', 'stime', 'subcategory', 'sum']
```
### Reads a CSV file into a DataFrame
```
df = pd.read_csv('combined_dataset.csv')

```
### shape :number of rows and coloumns 

```
df.shape

```
#### output 
```
(668522, 46)
```
### Analyze the Distribution of Attack vs Non-Attack Records
```
# Calculate percentage of attack and non-attack records
attack_percentages = df['attack'].value_counts(normalize=True) * 100

# Rename index values for readability (optional)
attack_percentages.index = ['Non-Attack' if val == 0 else 'Attack' for val in attack_percentages.index]

# Display the results
print(attack_percentages)
```
#### output 
```
Attack        99.928649
Non-Attack     0.071351
Name: proportion, dtype: float64
```
### Visualizing Attack vs Non-Attack Records (Bar Chart & Pie Chart)

```
import matplotlib.pyplot as plt
# Value counts and percentages
counts = df['attack'].value_counts()
labels = ['Non-Attack' if val == 0 else 'Attack' for val in counts.index]
percentages = (counts / counts.sum()) * 100

# Plot 1: Bar chart
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(labels, counts, color=['blue', 'green'])
plt.title("Count of Attack vs Non-Attack")
plt.ylabel("Number of Records")

# Plot 2: Pie chart
plt.subplot(1, 2, 2)
plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=['yellow', 'red'])
plt.title("Percentage of Attack vs Non-Attack")

plt.tight_layout()
plt.show()
```
#### output 
![download](https://github.com/user-attachments/assets/cef74a9a-1ad4-4d25-b60c-387bd99282b3)

####
