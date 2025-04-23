
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

###Exploring DoS and ReDoS Attack Patterns in the Dataset
```
# Preview category/subcategory values
print("Unique categories:", df['category'].unique())
print("Unique subcategories:", df['subcategory'].unique())

# Filter DoS and ReDoS records
dos_df = df[df['subcategory'].str.contains('DoS', case=False, na=False)]
redos_df = df[df['subcategory'].str.contains('ReDoS', case=False, na=False)]

# Print basic stats
print("Total DoS records:", len(dos_df))
print("Total ReDoS records:", len(redos_df))

# Compare features for DoS attacks
plt.figure(figsize=(10, 5))
sns.boxplot(data=dos_df[['pkts', 'bytes']], palette='coolwarm')
plt.title("DoS Attack Packet & Byte Distribution")
plt.ylabel("Value")
plt.show()

# Compare DoS to Non-Attack traffic
non_attack_df = df[df['attack'] == 0]
combined = pd.concat([
    dos_df.assign(type='DoS'),
    non_attack_df.assign(type='Non-Attack')
])

plt.figure(figsize=(12, 5))
sns.boxplot(x='type', y='pkts', data=combined)
plt.title("DoS vs Non-Attack: Packet Count")
plt.show()

# ReDoS-specific analysis
if len(redos_df) > 0:
    plt.figure(figsize=(10, 5))
    sns.histplot(redos_df['dur'], bins=30, kde=True, color='purple')
    plt.title("ReDoS Duration Distribution")
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.show()
else:
    print(" No ReDoS records found in the dataset.")
```
#### output 
```
Unique categories: ['DDoS' 'Normal' 'Reconnaissance' 'Theft']
Unique subcategories: ['UDP' 'Normal' 'OS_Fingerprint' 'Service_Scan' 'Data_Exfiltration'
 'Keylogging']
Total DoS records: 0
Total ReDoS records: 0

![download](https://github.com/user-attachments/assets/2730fab7-73ca-4174-a8e6-0cb4c66c080f)

No ReDoS records found in the dataset.
```
#### Data Cleaning and Preprocessing Steps
```
# 1. Standardize column names (strip spaces, lowercase, replace spaces with underscores)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Check and handle missing values
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Optional: Fill or drop based on strategy
# df.fillna(method='ffill', inplace=True)   # Forward fill
# df.dropna(inplace=True)                   # Drop rows with any NaN

# 3. Drop duplicates
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

# 4. Convert data types (if needed)
# For example, if 'dur' is object but should be float:
# df['dur'] = pd.to_numeric(df['dur'], errors='coerce')

# 5. Drop irrelevant columns (if any)
# Example: if there's a column like 'id' or unnamed index
df = df.loc[:, ~df.columns.str.contains('^unnamed')]

# 6. Check for consistent values in categorical columns
if 'category' in df.columns:
    print("Unique values in 'category':", df['category'].unique())
if 'subcategory' in df.columns:
    print("Unique values in 'subcategory':", df['subcategory'].unique())

# 7. Optional: encode categorical features for modeling
# df_encoded = pd.get_dummies(df, columns=['category', 'subcategory'], drop_first=True)

# 8. Final check
print("\nCleaned dataset shape:", df.shape)
print("Data types:\n", df.dtypes)

```
#### output 

```
Missing values:
 Series([], dtype: int64)
Removed 0 duplicate rows.
Unique values in 'category': ['DDoS' 'Normal' 'Reconnaissance' 'Theft']
Unique values in 'subcategory': ['UDP' 'Normal' 'OS_Fingerprint' 'Service_Scan' 'Data_Exfiltration'
 'Keylogging']

Cleaned dataset shape: (668522, 46)
Data types:
 ar_p_proto_p_dport                  float64
ar_p_proto_p_dstip                  float64
ar_p_proto_p_sport                  float64
ar_p_proto_p_srcip                  float64
n_in_conn_p_dstip                     int64
n_in_conn_p_srcip                     int64
pkts_p_state_p_protocol_p_destip      int64
pkts_p_state_p_protocol_p_srcip       int64
tnbpdstip                             int64
tnbpsrcip                             int64
tnp_pdstip                            int64
tnp_psrcip                            int64
tnp_perproto                          int64
tnp_per_dport                         int64
attack                                int64
bytes                                 int64
category                             object
daddr                                object
dbytes                                int64
dpkts                                 int64
dport                                object
drate                               float64
dur                                 float64
flgs                                 object
flgs_number                           int64
ltime                               float64
max                                 float64
mean                                float64
min                                 float64
pkseqid                               int64
pkts                                  int64
proto                                object
proto_number                          int64
rate                                float64
saddr                                object
sbytes                                int64
seq                                   int64
spkts                                 int64
sport                                object
srate                               float64
state                                object
state_number                          int64
stddev                              float64
stime                               float64
subcategory                          object
sum                                 float64
dtype: object
```
### Feature Selection and Importance Evaluation
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("combined_dataset.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Drop non-numeric and irrelevant columns (keep for EDA if needed)
df = df.select_dtypes(include=['int64', 'float64'])

# 1. Correlation with target
plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

# Display top features correlated with 'attack'
top_corr = cor['attack'].abs().sort_values(ascending=False)[1:11]
print("Top correlated features with 'attack':\n", top_corr)

# 2. Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
reduced_data = selector.fit_transform(df.drop('attack', axis=1))
selected_features = df.drop('attack', axis=1).columns[selector.get_support()]

print("\nLow-variance filter selected features:", list(selected_features))

# 3. Univariate selection (Chi2)
# Must be positive values for chi2
df_chi = df.fillna(0).copy()
X = df_chi.drop('attack', axis=1)
y = df_chi['attack']

# Apply chi-squared
selector_chi = SelectKBest(score_func=chi2, k=10)
selector_chi.fit(X, y)
chi_scores = pd.Series(selector_chi.scores_, index=X.columns)
chi_scores = chi_scores.sort_values(ascending=False)
print("\nTop features using Chi2:\n", chi_scores.head(10))

# 4. Feature importance using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[:10], y=importances.index[:10], palette='viridis')
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
```

#### output 

```
<ipython-input-14-bb48b4cee2d7>:9: DtypeWarning: Columns (20,38) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv("combined_dataset.csv")
```
![download](https://github.com/user-attachments/assets/582bd934-3bd4-4d91-b173-58f1e313de4c)
```
Top correlated features with 'attack':
 tnp_perproto     0.804492
tnbpsrcip        0.286662
tnp_psrcip       0.271961
tnbpdstip        0.257517
spkts            0.254837
sum              0.253827
tnp_pdstip       0.238696
pkts             0.232891
sbytes           0.228514
tnp_per_dport    0.212446
Name: attack, dtype: float64

e filter selected features: ['ar_p_proto_p_dport', 'ar_p_proto_p_dstip', 'ar_p_proto_p_sport', 'ar_p_proto_p_srcip', 'n_in_conn_p_dstip', 'n_in_conn_p_srcip', 'pkts_p_state_p_protocol_p_destip', 'pkts_p_state_p_protocol_p_srcip', 'tnbpdstip', 'tnbpsrcip', 'tnp_pdstip', 'tnp_psrcip', 'tnp_perproto', 'tnp_per_dport', 'bytes', 'dbytes', 'dpkts', 'drate', 'dur', 'flgs_number', 'ltime', 'max', 'mean', 'min', 'pkseqid', 'pkts', 'proto_number', 'rate', 'sbytes', 'seq', 'spkts', 'srate', 'state_number', 'stddev', 'stime', 'sum']

Top features using Chi2:
 tnbpsrcip       1.076362e+12
bytes           9.954825e+11
tnbpdstip       6.625210e+11
sbytes          6.282214e+11
dbytes          3.990383e+11
tnp_perproto    4.691692e+09
srate           1.191208e+09
dpkts           3.155281e+08
pkts            2.426495e+08
spkts           1.368482e+08
dtype: float64
<ipython-input-14-bb48b4cee2d7>:57: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=importances[:10], y=importances.index[:10], palette='viridis')




```
![download](https://github.com/user-attachments/assets/5218f1f8-46b9-46b8-9057-81aa875d214c)


### print some rows 

```
df.head()
```

### Random Forest Classification with Bagging and Model Evaluation
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Encode the target column
le = LabelEncoder()
y = le.fit_transform(df['attack'])

# Features (drop the target)
X = df.drop(columns=['attack'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bagging: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Random Forest (Bagging) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

```
#### output 

```
Random Forest (Bagging) Results:
Accuracy: 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       100
           1       1.00      1.00      1.00    133605

    accuracy                           1.00    133705
   macro avg       1.00      1.00      1.00    133705
weighted avg       1.00      1.00      1.00    133705

```

### XGBoost Classification: Model Training and Accuracy Evaluation

```
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Train Accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

# Test Accuracy
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("XGBoost Accuracy:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

```

#### output 

```
/usr/local/lib/python3.11/dist-packages/xgboost/core.py:158: UserWarning: [17:13:47] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)
XGBoost Accuracy:
Train Accuracy: 1.0000
Test Accuracy:  1.0000
```
