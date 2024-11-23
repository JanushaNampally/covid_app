import pandas as pd  # type: ignore
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
import joblib  # type: ignore

data = pd.read_csv("covid_final_dataset_150.csv")

X = data[['Age', 'Fever', 'Cough', 'Hypertension', 'Diabetes', 'TravelHistory', 'LossOftasteSmell', 'ShortnessOfBreath', 'CloseContact']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100: 2f}%')

joblib.dump(model, '../analysis/covid_rf_model.pkl')
print("Model trained and saved to '../analysis/covid_rf_model.pkl'")