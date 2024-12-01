import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = {
    "Category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Values": [10, 12, 14, 20, 22, 19, 30, 29, 32, 31]
}
df = pd.DataFrame(data)
sns.kdeplot(data=df,x="Category",hue="Values",multiple="stack")
plt.show()