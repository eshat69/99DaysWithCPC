from sklearn.impute import SimpleImputer
import numpy as np
data = np.array([
    [1, 2, "nan"],
    [4, "nan", 6],
    ["nan", 8, 9]
])
imputer = SimpleImputer(strategy="mean")
imputed_data = imputer.fit_transform(data)

print("Original Data:")
print(data)
print("\nImputed Data:")
print(imputed_data)
