import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Notebook 2 references
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

data = pd.read_csv('powerpredict.csv')


# Transforming categorical values to numerical values; Using pandas get_dummies for one-hot encoding
"""
The pandas method get_dummies() can take various features as input and then for each feature, it creates a column for each of the feature's categories.
These columns will have 0/1 entries, indicating if the specified category is active in the corresponding row.
"""
data = pd.get_dummies(data, columns=['Bedrock_weather_main', 'Bedrock_weather_description', 'Gotham City_weather_main', 'Gotham City_weather_description',
                                     'New New York_weather_description', 'New New York_weather_main',
                                     'Springfield_weather_description', 'Springfield_weather_main',
                                     'Paperopoli_weather_main', "Paperopoli_weather_description"], drop_first=True)

# Prepare features and target
X = data.drop('power_consumption', axis=1).values
y = data['power_consumption'].values

"""
plt.plot(X, y, color='red', label='true')
plt.grid()
plt.legend()
# Verify the data types to ensure all are numeric
print(data.dtypes)

print(plt)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""

