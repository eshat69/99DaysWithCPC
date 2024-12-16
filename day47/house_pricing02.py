import pandas as pd
from datetime import datetime

# Sample data (replace with your actual dataset)
data = {
    'year_built': [2000, 1995, 2010],
    'sale_price': [300000, 250000, 400000],
    'square_footage': [2000, 1800, 2200]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate current year
current_year = datetime.now().year

# Calculate age of the house
df['age_of_house'] = current_year - df['year_built']

# Calculate price per square foot
df['price_per_sqft'] = df['sale_price'] / df['square_footage']

# Display the results
print(df)
