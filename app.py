from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = load_model('seizure_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define categories
categories = ['Normal', 'Preictal', 'Seizure']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="⚠️ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="⚠️ No file selected!")

    try:
        # Load EEG data from uploaded .txt file
        data = np.loadtxt(file)
        data = np.array(data).reshape(1, -1)

        # Apply same scaling as training
        data_scaled = scaler.transform(data)
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error reading or scaling data: {str(e)}")

    # Predict
    prediction = model.predict(data_scaled)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    result = f"🧠 Prediction: {categories[predicted_class]} ({confidence:.2f}% confidence)"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
