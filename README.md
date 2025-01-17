# Energy-Optimizer

Creating an `Energy-Optimizer` tool that uses machine learning to analyze and optimize household energy consumption involves various steps such as data collection, preprocessing, model training, and prediction. For simplicity, I'll provide a Python program that uses simulated data to demonstrate this process. The code will use the `scikit-learn` library for model training.

### Requirements:
1. Python 3.x
2. Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib` (for plotting results)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_synthetic_data(num_samples=1000):
    """Generates synthetic energy consumption data."""
    np.random.seed(42)
    data = {
        'temperature': np.random.uniform(0, 40, num_samples),           # Temperature in Celsius
        'humidity': np.random.uniform(20, 80, num_samples),             # Humidity in percentage
        'num_occupants': np.random.randint(1, 6, num_samples),          # Number of occupants
        'day_of_week': np.random.randint(0, 7, num_samples),            # Day of the week
        'time_of_day': np.random.randint(0, 24, num_samples),           # Hour of the day
        'past_energy_consumption': np.random.uniform(300, 700, num_samples) # kWh
    }

    # Simulate energy consumption with some random noise
    data['energy_consumption'] = (
        0.5 * data['temperature'] +
        0.8 * data['humidity'] +
        1.2 * data['num_occupants'] -
        0.3 * data['day_of_week'] +
        0.1 * data['time_of_day'] +
        0.05 * data['past_energy_consumption'] +
        np.random.normal(0, 3, num_samples)
    )

    return pd.DataFrame(data)

# Load and split the data
def load_and_split_data():
    """Loads synthetic data and splits it into train and test sets."""
    df = generate_synthetic_data()
    X = df.drop('energy_consumption', axis=1)
    y = df['energy_consumption']
    return train_test_split(X, y, test_size=0.2, random_state=42)

class EnergyOptimizer:
    def __init__(self):
        """Initializes the EnergyOptimizer with a RandomForestRegressor model."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """Trains the RandomForest model."""
        try:
            self.model.fit(X_train, y_train)
            print("Model training completed.")
        except Exception as e:
            print(f"An error occurred during model training: {e}")

    def evaluate(self, X_test, y_test):
        """Evaluates the model's performance on test data."""
        try:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Mean Squared Error: {mse:.2f}")
            return y_pred
        except Exception as e:
            print(f"An error occurred during model evaluation: {e}")

    def plot_predictions(self, y_test, y_pred):
        """Plots the actual vs. predicted energy consumption."""
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Actual')
        plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='Predicted')
        plt.title('Actual vs. Predicted Energy Consumption')
        plt.xlabel('Sample Index')
        plt.ylabel('Energy Consumption (kWh)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Initialize and train the EnergyOptimizer
    optimizer = EnergyOptimizer()
    optimizer.train(X_train, y_train)

    # Evaluate the model
    y_pred = optimizer.evaluate(X_test, y_test)

    # Plot predictions
    optimizer.plot_predictions(y_test, y_pred)
```

### Explanation:
- **Synthetic Data Generation:** The program generates synthetic data representing various factors contributing to household energy consumption.
- **Data Splitting:** The data is split into training and testing datasets to evaluate model performance.
- **Modeling:** The program uses a `RandomForestRegressor` as the machine learning model.
- **Training and Evaluation:** The model is trained on the training dataset and evaluated on the test dataset, calculating the Mean Squared Error (MSE) as a performance metric.
- **Plotting:** It visualizes actual vs. predicted energy consumption using a scatter plot.

This program should be viewed as a basic illustration. For real-world applications, you would replace the synthetic data with real energy consumption data and potentially use more sophisticated models and feature engineering techniques to improve accuracy and optimization recommendations. Error handling is incorporated to catch potential issues during model training and evaluation.