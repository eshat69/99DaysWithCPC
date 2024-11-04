import pandas as pd
data = {
    'Temperature': [30, 35, 28, 40, 33, 29],
    'Humidity': [70, 65, 80, 60, 75, 78],
    'Sales': [200, 250, 180, 300, 240, 210]
}
df = pd.DataFrame(data)
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)
