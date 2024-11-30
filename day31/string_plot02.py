import matplotlib.pyplot as plt
import seaborn as sns
import pandas as p
data = {
    "Category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Values": [10, 12, 14, 20, 22, 19, 30, 29, 32, 31]
}
sns.stripplot(data=data, x="Category", y="Values", jitter=True, palette="pastel")
plt.show()