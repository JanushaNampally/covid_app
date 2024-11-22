import pandas as pd 
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
import joblib  # type: ignore

data = pd.read_csv(r"scripts/covid_binary_dataset.csv")

X = data[['feature1', 'feature2', 'feature3']]
y = data['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, '/analysis/covid_rf_model.pkl')
print("Model trained and saved to '/analysis/covid_rf_model.pkl'")