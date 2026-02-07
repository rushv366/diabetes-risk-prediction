from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


model = joblib.load("model/diabetes_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")  


@app.route("/predict", methods=["POST"])
def predict():
   
    try:
        input_data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # Convert to numpy array for model
        input_array = np.array([input_data])

        # Make prediction
        prediction = model.predict(input_array)

        if prediction[0] == 1:
            result = "High risk of diabetes"
        else:
            result = "Low risk of diabetes"

        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return f"Error: {e}"

# 5. Run the app
if __name__ == "__main__":
    app.run(debug=True)
