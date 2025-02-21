# EXPERIMENT 1
<br> This repository contains the code and results for the first experiment in the Data Science Fundamentals with Python course. The objective of this experiment is to load and merge datasets obtained from the UCI Machine Learning Repository.

Steps-
Loading Datasets: The datasets are loaded using the Pandas library from the UCI ML Repository.
Inspecting Datasets: The first few rows of each dataset are displayed to understand their structure.
Merging Datasets: The datasets are merged on a common column (e.g., 'ID').
Saving the Merged Dataset: The merged dataset is optionally saved as a CSV file.


Concepts Used-
Pandas Library: Used for data manipulation and analysis.
DataFrames: Data structures provided by Pandas to store and manipulate tabular data.
Merging Datasets: The pd.merge() function is used to combine two datasets based on a common key.

<br>

```
import pandas as pd

df1 = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", delimiter= ';')  # Replace with actual file path
df2 = pd.read_csv("/content/drive/MyDrive/winequality-white.csv",delimiter= ';')
print("red wine data: ")
df1['type']='red'
print(df1.head())
print("white wine data: ")
df2['type']='white'
print(df2.head())
merged= pd.concat([df1,df2],ignore_index= True)


print("combined: ")
print(merged.head())

```
<br>

