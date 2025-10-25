from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model (must output 3 probabilities)
model = load_model('seizure_model.h5')

# Define category labels
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
        # Load EEG data from .txt file
        data = np.loadtxt(file)
        data = np.array(data).reshape(1, -1)  # reshape for model input
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error reading file: {str(e)}")

    # Predict
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction)  # get index of max probability
    confidence = prediction[0][predicted_class] * 100

    result = f"🧠 Prediction: {categories[predicted_class]} ({confidence:.2f}% confidence)"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
