import joblib # type: ignore
from django.shortcuts import render # type: ignore


def predictor_view(request):
    
    # If the request is POST, handle the form submission
    if request.method == "POST":
        try:
            age = int(request.POST.get('age', 0))
            fever = int(request.POST.get('fever', 0))
            cough = int(request.POST.get('cough', 0))
            hypertension = int(request.POST.get('hypertension', 0))
            diabetes = int(request.POST.get('diabetes', 0))
            travel_history = int(request.POST.get('travel_history', 0))
            loss_of_taste_smell = int(request.POST.get('loss_of_taste_smell', 0))
            shortness_of_breath = int(request.POST.get('shortness_of_breath', 0))
            close_contact = int(request.POST.get('close_contact', 0))
            
            model = joblib.load('covid_app/analysis/covid_rf_model.pkl')
            input_features = [[age, fever, cough, hypertension, diabetes, travel_history, loss_of_taste_smell, shortness_of_breath, close_contact]]
            
            prediction = model.predict(input_features)
            #result = "You are likely COVID positive." if prediction[0] == 1 else "You are likely COVID negative."
        except Exception as e:
            prediction = f"Error occured: {str(e)}"
        return render(request, 'analysis/predictor.html', {'prediction': prediction})
    return render(request, 'analysis/predictor.html')
    # Pass prediction and error_message only after form submission
    