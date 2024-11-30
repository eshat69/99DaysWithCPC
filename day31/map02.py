import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Example data
data = {
    "Category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
    "Values": [10, 12, 14, 20, 22, 19, 30, 29, 32, 31]
}
df = pd.DataFrame(data)
# Display a color palette
sns.set_palette("viridis")
palette = sns.color_palette("viridis", as_cmap=False)
sns.palplot(palette)  # This works in older versions of Seaborn

# FacetGrid
g = sns.FacetGrid(df, col="Category", height=4, sharey=True)
g.map(sns.barplot, "Category", "Values", order=["A", "B", "C"], ci=None)
plt.show()
