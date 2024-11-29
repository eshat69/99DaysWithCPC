import matplotlib.pyplot as plt
import seaborn as sns
# Example data with 'Ethnicity' added
data = {
    "day": [
        1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6, 7
    ],
    "nop": [
        50, 40, 45, 20, 20, 59, 25,
        60, 35, 55, 30, 25, 70, 20,
        80, 65, 30, 50, 45, 75, 40,
        55, 60, 35, 70, 25, 85, 30
    ],
    "Ethnicity": [
        "Group A", "Group B", "Group A", "Group B", "Group A", "Group B", "Group A",
        "Group C", "Group C", "Group C", "Group C", "Group C", "Group C", "Group C",
        "Group D", "Group D", "Group D", "Group D", "Group D", "Group D", "Group D",
        "Group A", "Group C", "Group B", "Group B", "Group A", "Group A", "Group A"
    ]
}

print(data)

# Create a scatter plot
sns.scatterplot(data=data, x="day", y="nop", hue="Ethnicity", style="Ethnicity", palette="deep")
plt.title("Scatter Plot of Number of People (nop) by Day")
plt.xlabel("Day")
plt.ylabel("Number of People (nop)")
plt.show()
"""x="day" and y="nop": Specifies the data columns for the x and y axes.
hue="Ethnicity": Colors the points based on the Ethnicity column.
style="Ethnicity": Differentiates groups with different marker styles.
palette="deep": Sets a color palette for better visualization.
Added Titles and Labels: To make the chart more informative."""
