# Created by: Addisu Taye
# Date: 28-JUN-2025
# Purpose: This Flask API serves as an endpoint for predicting the sentiment
#          of movie review texts. It leverages the pre-trained sentiment
#          analysis model and preprocessing logic defined in `src/predict.py`.
# Key Features:
#   - Exposes a POST endpoint `/predict` for sentiment inference.
#   - Accepts JSON input containing the text to be analyzed.
#   - Returns the predicted sentiment (positive/negative) and a confidence score
#     in JSON format.
#   - Integrates the advanced text preprocessing and model prediction pipeline.
#   - Serves a simple web-based user interface for easy interaction.

from flask import Flask, request, jsonify, render_template # Added render_template
from src.predict import predict_sentiment 

# Initialize the Flask application
app = Flask(__name__)

# New Route: Serve the main HTML page
@app.route('/')
def index():
    """
    Renders the main index.html page, which provides the user interface
    for entering text and getting sentiment predictions.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the /predict endpoint.
    It expects a JSON payload with a 'text' key containing the movie review.
    
    Example JSON request body:
    {
        "text": "This movie was absolutely fantastic!"
    }
    
    Returns:
        JSON response containing the predicted sentiment ('positive' or 'negative')
        and the confidence score.
        Example: {'sentiment': 'positive', 'confidence': 95.5}
    """
    data = request.get_json()

    if not data:
        return jsonify({
            'error': 'Invalid JSON or missing data in request body.',
            'message': 'Please send a JSON object with a "text" key.'
        }), 400 

    text = data.get('text', '').strip() 

    if not text:
        return jsonify({
            'error': 'Text field is empty.',
            'message': 'The "text" field in the JSON request cannot be empty.'
        }), 400 

    sentiment, confidence = predict_sentiment(text)

    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)