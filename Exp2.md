# EXPERIMENT 2
<br>
Steps-
<br>
1.Loading datasets: The datasets are loaded using the Pandas library from the UCI ML Repository.
<br>
2.Handling Missing Values:Handle missing values by removing them or replacing Null/NaN with defaults, statistics, or predictions.
<br>
3.Removing Duplicates: Remove duplicates by dropping them or keeping the first/last occurrence based on data needs.
<br>
4.Handling Outliers:Handle outliers by removing or capping values beyond the IQR range (1.5×IQR rule).
<br>

Concepts Used-
<br>
Pandas Library: Used for data manipulation and analysis.
<br>
DataFrames: Data structures provided by Pandas to store and manipulate tabular data.
<br>
Data Cleaning: Techniques such as removing duplicates, handling missing values, and standardizing data.
<br>

```
   
import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/movies.csv')
#data cleaning:
print("checking for null values(is null)")
df1.isnull()
print("checking for null values(not null)")
df1.notnull()
print("checking for duplicated values: ")
df1.duplicated()
print("after dropping duplicated values: ")
df.drop_duplicates()
#handling outliers
df.columns = df.columns.str.strip()
print("Column Names:", df.columns)

q1 = df['RATING'].quantile(0.25)
q3 = df['RATING'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df_clean = df[(df['RATING'] >=lower_bound) & (df['RATING'] <= upper_bound)]
df_clean.to_csv("/content/drive/MyDrive/cleaned_movies.csv", index=False)
print(" Outliers removed! Cleaned data saved as 'cleaned_movies.csv'.")
```
