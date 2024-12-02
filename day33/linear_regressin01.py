import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
data = {
    "age": [25, 30, 35, 40, 45],
    "premium": [18000, 32000, 40000, 47000, 55000]
}
df = pd.DataFrame(data)
print(df)
sns.lmplot(x="age", y="premium", data=df)
plt.title("Age vs Premium with Regression Line")
plt.show()
# Step 3: Fit a Linear Regression model
reg = LinearRegression()
reg.fit(df[['age']], df['premium'])
# Step 4: Display the model parameters
print("Coefficient (Slope):", reg.coef_[0])
print("Intercept:", reg.intercept_)
# Step 5: Make predictions (optional)
predicted_premiums = reg.predict(df[['age']])
print("Predicted Premiums:", predicted_premiums)
