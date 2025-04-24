
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
### Libraries used and Merging and Loading the dataset  
```
import pandas as pd
import glob


# Specify the CSV file paths directly
csv_files = [
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_1.csv',
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_2.csv',
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_3.csv',
    '/content/drive/MyDrive/UNSW_2018_IoT_Botnet_Full5pc_4.csv'
]


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

### balancing the data 
```
import pandas as pd
import numpy as np

# Simulating 'combined_df' based on your sample
# Assuming your dataset has 3668522 non-attack records and no attack records
# First, let's create some synthetic data for attack cases

# Define number of attack records (synthetic)
num_attack = 1000  # For example, let's create 1000 attack records

# Generate synthetic 'attack' data (random values for illustration purposes)
attack_data = {
    'pkts': np.random.randint(1, 100, size=num_attack),
    'bytes': np.random.randint(100, 5000, size=num_attack),
    'dur': np.random.uniform(0.01, 10, size=num_attack),
    'attack': np.ones(num_attack)  # Label as '1' for attack
}

# Convert to DataFrame
attack_df = pd.DataFrame(attack_data)

# Simulate the non-attack data (already in your dataset)
non_attack_df = combined_df[combined_df['attack'] == 0].sample(num_attack, random_state=42)

# Combine both attack and non-attack data
combined_balanced_df = pd.concat([attack_df, non_attack_df])

# Shuffle the combined DataFrame
combined_balanced_df = combined_balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the new balance of the dataset
print(combined_balanced_df['attack'].value_counts())

# Visualize the distribution after balancing
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(x='attack', data=combined_balanced_df, palette='Set2')
plt.title("Balanced Attack vs Non-Attack Distribution")
plt.xlabel("Attack (0 = Non-Attack, 1 = Attack)")
plt.ylabel("Count")
plt.show()
```
#### output 



![download (1)](https://github.com/user-attachments/assets/8180f8b3-bf97-4733-9058-f4697b14a63e)




### Exploring DoS and ReDoS Attack Patterns in the Dataset
```
# Preview category/subcategory values
print("Unique categories:", df['category'].unique())
print("Unique subcategories:", df['subcategory'].unique())

# Filter ![download](https://github.com/user-attachments/assets/0c9bc2cd-3a24-448b-9ba7-bea058919c0d)
DoS and ReDoS records
dos_df = df[df['subcategory'].str.contains('DoS', case=False, na=False)]
redos_df = df[df['subcategory'].str.contains('ReDoS', case=False, na=False)]

# Print basic stats
prin![download](https://github.com/user-attachments/assets/802615fa-b29c-4e3f-84d8-fbc4aca16981)
t("Tot![download](https://github.com/user-attachments/assets/f8c689f5-fad1-453c-94bf-078747457658)
al DoS records:", len(dos_df))
print("Total ReDoS records:", len(redos_df))

# Compare features for DoS attacks
plt.figure(figsize=(10, 5))
sns.boxplot(data=dos_df[['pkts', 'bytes']], palette='coolwarm')
plt.title("DoS![download](https://github.com/user-attachments/assets/4e94d8b5-d5bd-4405-b64f-8946b745a201)
 Attack Packet & Byte Distribution")
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
```
![download](https://github.com/user-attachments/assets/2730fab7-73ca-4174-a8e6-0cb4c66c080f)
![download](https://github.com/user-attachments/assets/5b32dee0-0e23-4ae6-90c3-44892581621d)

```
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
![download](https://github.com/user-attachments/assets/1792e3f4-41c9-441c-aacc-36b133f93413)
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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Assuming 'combined_balanced_df' is the balanced dataset you created earlier

# Define features and target variable
features = ['pkts', 'bytes', 'dur']  # Adjust or add more features if necessary
X = combined_balanced_df[features]
y = combined_balanced_df['attack']

#  Step 1: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#  Step 2: Feature Selection - Filter Method (SelectKBest)
filter_selector = SelectKBest(score_func=f_classif, k=2)  # Select top 2 features based on ANOVA F-test
X_train_filter = filter_selector.fit_transform(X_train, y_train)
X_test_filter = filter_selector.transform(X_test)

# Get selected feature indices
selected_filter_features = filter_selector.get_support(indices=True)

#  Step 3: Feature Selection - Embedded Method (Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances and select top 2 important features
importances = rf.feature_importances_
top_n = 4
top_n_indices = importances.argsort()[-top_n:]

# Combine selected features from both methods
selected_features = np.union1d(selected_filter_features, top_n_indices)

#  Step 4: Create New Training and Test Data with Selected Features
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]

# Step 5: Scale Features (Important for algorithms like Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

#  Step 6: Train the Final Model (RandomForestClassifier)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

#  Step 7: Make Predictions and Evaluate the Model
y_pred = clf.predict(X_test_scaled)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#  Step 8: Feature Importance from Random Forest (for interpretation)
plt.figure(figsize=(8, 6))
sns.barplot(x=rf.feature_importances_[top_n_indices], y=[features[i] for i in top_n_indices])
plt.title("Top 3 Important Features Based on Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()

```

#### output 

```
Classification Report:

              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00       200
         1.0       1.00      0.99      0.99       200

    accuracy                           0.99       400
   macro avg       1.00      0.99      0.99       400
weighted avg       1.00      0.99      0.99       400
```
![download (2)](https://github.com/user-attachments/assets/03259d99-250f-4b44-a0b6-8239bf02547e)

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

### confusion matrix and roc curve 

```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Confusion Matrix with class names
def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    # Define labels if your target has binary classes: 0 = Normal, 1 = Attack
    labels = ["Normal", "Attack"]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    plt.show()

# ROC Curve
def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    # Get probability scores or decision function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Call the functions
plot_confusion_matrix(y_test, y_pred, model_name="Random Forest")
plot_roc_curve(model, X_test, y_test, model_name="Random Forest")
```
#### output 

![download (3)](https://github.com/user-attachments/assets/82ed211e-c3dd-4a81-a934-89936385af7f)
![download (4)](https://github.com/user-attachments/assets/743255c0-3367-4ca3-a835-2fa7376088ee)
