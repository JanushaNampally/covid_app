import joblib

# Load the model (make sure to provide the correct path to your .pkl file)
model = joblib.load('path/to/covid_rf_model.pkl')

# Test with some example input (adjust the values based on your model's feature expectations)
input_data = [[30, 1, 0]]  # Example: Age 30, Fever (1 = Yes), Cough (0 = No)

# Make the prediction
prediction = model.predict(input_data)

# Print the result
print("Prediction:", prediction)  # Should output either 0 or 1 (or another class depending on your model)
