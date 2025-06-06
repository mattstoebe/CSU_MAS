import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),
    'feature4': np.random.normal(0, 1, n_samples),
    'feature5': np.random.normal(0, 1, n_samples)
}

# Create target variable with some relationship to features
data['target'] = (
    2 * data['feature1'] +
    1.5 * data['feature2'] +
    -1 * data['feature3'] +
    0.5 * data['feature4'] +
    -0.5 * data['feature5'] +
    np.random.normal(0, 0.5, n_samples)
)

# Convert to DataFrame
df = pd.DataFrame(data)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# Save the model
joblib.dump(model, 'model.joblib')

print("Model trained and saved as 'model.joblib'")
print("\nYou can now use this model with the Streamlit app.")
print("The model expects the following features:")
for feature in X.columns:
    print(f"- {feature}")

# Save a sample input file for testing
sample_input = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 5),
    'feature2': np.random.normal(0, 1, 5),
    'feature3': np.random.normal(0, 1, 5),
    'feature4': np.random.normal(0, 1, 5),
    'feature5': np.random.normal(0, 1, 5)
})

sample_input.to_csv('sample_input.csv', index=False)
print("\nA sample input file 'sample_input.csv' has been created for testing.") 