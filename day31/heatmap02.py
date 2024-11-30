import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#  data dictionary
data = {
    "name": ['sdf', 'dfwrg', 'sfg', 'erfg', 'sdf', 'dfwrg', 'sfg', 'erfg',
             'sdf', 'dfwrg', 'sfg', 'erfg', 'sdf', 'dfwrg', 'sfg', 'erfg',
             'sdf', 'dfwrg', 'sfg', 'erfg', 'sdf', 'dfwrg', 'sfg', 'erfg',
             'sdf', 'dfwrg', 'sfg', 'erfg', 'sdf', 'dfwrg', 'sfg', 'erfg'],
    "date": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             1, 2, 4, 1, 2, 4, 245, 24, 23, 3, 22, 55, 88, 56, 78, 43]
}
df = pd.DataFrame(data)
# Create a histogram for name frequencies
sns.histplot(data=df, x="name", shrink=0.8, color="purple")

plt.title("Frequency of Names")
plt.xlabel("Name")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

"""Scatter Plot: Use sns.scatterplot for relationships between name and date.
Histogram: Use sns.histplot to display frequency distributions.
Markers and Other Parameters: Removed unsupported parameters from histplot."""
