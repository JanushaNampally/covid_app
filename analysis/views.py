from django.shortcuts import render
import joblib

def predictor_view(request):
    prediction = None
    error_message = None
    
    # If the request is POST, handle the form submission
    if request.method == "POST":
        try:
            feature1 = float(request.POST.get('feature1', 0))
            feature2 = float(request.POST.get('feature2', 0))
            feature3 = float(request.POST.get('feature3', 0))
        
            model = joblib.load('analysis/covid_rf_model.pkl')
            prediction = model.predict([[feature1, feature2, feature3]])[0]
        
        except ValueError:
            error_message = "Invalid input. Please enter valid numeric values."
        except FileNotFoundError:
            error_message = "Model file not found. Ensure 'covid_rf_model.pkl' exists in the specified path."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
    
    # Pass prediction and error_message only after form submission
    return render(request, 'analysis/predictor.html', {'prediction': prediction, 'error_message': error_message})
