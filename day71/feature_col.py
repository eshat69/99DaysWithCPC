import tensorflow as tf
import pandas as pd  # Assuming you have pandas imported for DataFrame operations
# Example DataFrame (replace with your actual DataFrame)
dftrain = pd.DataFrame({
    'sex': ['male', 'female', 'male', 'female'],
    'n_siblings_spouses': [1, 2, 0, 1],
    'parch': [0, 1, 2, 1],
    'class': ['First', 'Second', 'Third', 'First'],
    'deck': ['A', 'B', 'C', 'D'],
    'embark_town': ['Southampton', 'Cherbourg', 'Southampton', 'Cherbourg'],
    'alone': ['n', 'n', 'y', 'n'],
    'age': [22, 38, 26, 35],
    'fare': [7.25, 71.28, 8.05, 53.1]
})
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []
# Creating categorical feature columns
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # Get unique values for each categorical feature
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
# Creating numeric feature columns
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)

