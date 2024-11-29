import matplotlib.pyplot as plt
import seaborn as sns
# Example data with 'Ethnicity' added
data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "nop": [50, 40, 45, 20, 20, 59, 25],
    "Ethnicity": ["Group A", "Group B", "Group A", "Group B", "Group A", "Group B", "Group A"]
}
print(data)

# Create the line plot with hue
sns.lineplot(data=data, x="day", y="nop", hue="Ethnicity", palette="deep")

# Display the plot
plt.show()

