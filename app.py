# app.py
from flask import Flask, request, render_template, jsonify
import os
import sys

# Add the current directory to the path for module imports
# This is usually needed for local development and can help with Vercel too
# if your src folder is not directly in the path for imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import your prediction function
from src.predict import predict_sentiment # Assuming predict_sentiment is in src/predict.py

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Route for the home page (where your HTML UI is)
@app.route('/')
def home():
    return render_template('index.html') # Ensure you have an index.html in your templates folder

# Route for the prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        sentiment, confidence = predict_sentiment(text)
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Prediction error: {e}") # Log error for debugging in Vercel logs
        return jsonify({'error': 'An internal error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)