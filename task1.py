# for data manipulation
import pandas as pd

# for numerical operations
import numpy as np

# for visualization and outlier detection
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv ('Titanic-Dataset.csv')

# Drop the 'Cabin' column because it has too many missing values to be useful
df = df.drop (columns=['Cabin'])

# Fill missing 'Age' values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' values with the most frequent value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Verify there are no missing values left
print ("\nMissing Values After Cleaning:")
print (df.isnull().sum())

# View the first 5 rows to understand the structure and contents
print ("First 5 Rows:")
print (df.head())

# Label Encoding for 'Sex' (converts 'male'/'female' to 1/0)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for 'Embarked' (creates separate columns for each port)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Visualize outliers using a boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Passenger Fares')
plt.show()

# Standardize the 'Age' and 'Fare' columns (centers data to mean 0, standard deviation 1)
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# View the final processed dataset
print("\nFinal Processed Dataset:")
print(df.head())

# Remove extreme outliers (e.g., fares over 300)
df = df[df['Fare'] < 300]
print(f"\nRemaining rows after outlier removal: {len(df)}")

# Check the data types of each column and total non-null values
print ("\nBasic Info & Data Types:")
print (df.info())

# Count the exact number of missing (null) values in each column
print ("\nMissing Values Count:")
print (df.isnull().sum())