from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("E:\data\Desktop\python\ML\student_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    prediction = model.predict([[hours, attendance]])[0]

    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    return render_template('index.html', prediction_text=f"The student will likely: {result}")

if __name__ == "__main__":
    app.run(debug=True)
