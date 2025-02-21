# EXPERIMENT 3 
<br>
Introduction to Encoding Techniques
<br>
1. Encoding techniques are essential for converting categorical variables into numerical format for use in machine learning algorithms. Two common encoding techniques used in this experiment are:
<br>
2. One-Hot Encoding: Converts categorical values into a set of binary columns where each unique category is represented by a binary column. This is ideal for nominal data without an inherent order.
<br>
Ordinal Encoding: Assigns unique integers to categories with an inherent order. This method is best suited for ordinal data where categories have a meaningful sequence.
<br>
Concepts Used-
<br>
Pandas Library: For data manipulation and analysis.
One-Hot Encoding: Converting categorical data to binary columns.
Ordinal Encoding: Encoding ordinal data into integer values based on a predefined order.
<br>

```

#1. label encoding
from sklearn.preprocessing import LabelEncoder
data=['Low','High','Medium','High','Medium']
encoder= LabelEncoder()
encoded_data= encoder.fit_transform(data)
print(f"Label encoded data: {encoded_data}")
#2. one hot encoding
import pandas as pd
data=['Red','Blue','Green','Blue','Red']
df= pd.DataFrame(data,columns=['Color'])
one_hot_encoded=pd.get_dummies(df['Color'])
print("one hot encoded: \n")
print(one_hot_encoded)
#3. ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
data=[['Low'],['High'],['Medium'],['High'],['Medium']]
encoder= OrdinalEncoder(categories=[['Low','Medium','High']])
encoded_data=encoder.fit_transform(data)
print(f"Ordinal Encoded Data: {encoded_data}")
#4. Target encoding
!pip install category_encoders
import pandas as pd
import category_encoders as ce
data= {'Color':['Red','Blue','Green','Blue','Red','Blue','Green','Green','Green','Blue'],'Target':['1','0','0','1','1','1','0','1','0','1']}
df=pd.DataFrame(data)
df['Target'] = df['Target'].astype(int)
encoder= ce.TargetEncoder(cols=['Color'])
encoded_data= encoder.fit_transform(df['Color'],df['Target'])
print(f"Target encoded: {encoded_data}")
#5. binary encoding
import category_encoders as ce
data=['Red','Green','Blue','Red','Grey']
encoder = ce.BinaryEncoder(cols=['Color'])
encoded_data= encoder.fit_transform(pd.DataFrame(data,columns=['Color']))
print("binary encoded: \n")
print(encoded_data)
#6. frequency encoding
import pandas as pd
data=['Red','Green','Blue','Red','Red']
series_data= pd.Series(data)
frequency_encoding= series_data.value_counts()
encoded_data= [frequency_encoding[x] for x in data]
print("frequency encoded: \n")
print("encoded data: ",encoded_data)

```
<br>

