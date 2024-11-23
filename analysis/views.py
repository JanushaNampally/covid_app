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
            cardiovascular = int(request.POST.get('cardiovascular', 0))
            obesity = int(request.POST.get('obesity', 0))
            
            model = joblib.load('covid_app/analysis/covid_rf_model.pkl')
            input_features = [[age, fever, cough, hypertension, cardiovascular, obesity]]
            
            prediction = model.predict(input_features)
            result = "You are likely COVID positive." if prediction[0] == 1 else "You are likely COVID negative."
        except Exception as e:
            result = f"Error occured: {str(e)}"
        return render(request, 'predictor.html', {'result': result})
    return render(request, 'predictor.html')
    # Pass prediction and error_message only after form submission
    