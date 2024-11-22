from django.shortcuts import render  # type: ignore
import joblib # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import numpy as np 
from django.shortcuts import render # type: ignore

# Create your views here.  # type: ignore
def predictor_view(request):
    prediction = None 
    
    error_message = None 
    if request.method == "POST":
        try: 
            feature1 = float(request.POST.get('feature1', 0))
            feature2 = float(request.POST.get('feature2', 0))
            feature3 = float(request.POST.get('feature3', 0))
        
            model = joblib.load('analysis\covid_rf_model.pkl')
            print(model)
            prediction = model.predict([[feature1, feature2, feature3]])[0]
        
        
        except ValueError:
            error_message = "Invalid input. Please enter valid numeric values."
        except FileNotFoundError:
            error_message = "model file not found. ENsure 'covid_rf_model.pkl' exists in the specified path."
        
        except Exception as e:
            error_message = f"An error occured: {str(e)}"
            print(error_message)
            print(e)
        
    return render(request, 'analysis/predictor.html', {'prediction': prediction, 'error_message': error_message})



