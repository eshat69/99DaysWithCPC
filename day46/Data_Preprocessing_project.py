import pandas as pd
import numpy as np

# Create a sample dataframe
data = {
    'id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, np.nan, 40],
    'income': [50000, 60000, 75000, 80000, 90000],
    'score': [85, 88, 82, 90, 75]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)
print()

# 1. Cleaning Data
# Assume we want to remove rows with missing 'age' values and outliers in 'income'
df_cleaned = df.dropna(subset=['age'])
df_cleaned = df_cleaned[df_cleaned['income'] < 90000]

print("Cleaned Data:")
print(df_cleaned)
print()

# 2. Handling Missing Values
# Fill missing 'age' values with mean
mean_age = df['age'].mean()
df_cleaned['age'].fillna(mean_age, inplace=True)

print("Data after handling missing values:")
print(df_cleaned)
print()

# 3. Smoothing Data
# Smooth 'score' using a moving average (window size of 2)
df_cleaned['smoothed_score'] = df_cleaned['score'].rolling(window=2, min_periods=1).mean()

print("Data after smoothing 'score' column:")
print(df_cleaned)
print()

# 4. Normalizing or Scaling Data
# Min-max scaling 'income'
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_cleaned['scaled_income'] = scaler.fit_transform(df_cleaned[['income']])

print("Data after scaling 'income' column:")
print(df_cleaned)
