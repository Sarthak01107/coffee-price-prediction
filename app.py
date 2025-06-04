import pandas as pd
from prophet import Prophet
import pickle
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Define the path to the saved model
MODEL_PATH = 'prophet_model.pkl'

# Load the trained Prophet model when the Flask app starts
model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Prophet model loaded successfully from {MODEL_PATH}.")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Please run model.py first to train and save the model.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Ensure model is None if loading fails

@app.route('/')
def home():
    """
    Renders the index.html page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """
    API endpoint to get time series predictions.
    Expects a 'days' parameter in the query string.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure 'prophet_model.pkl' exists and is accessible."}), 500

    try:
        # Get the number of days to predict from the query parameters
        days = int(request.args.get('days', 7)) # Default to 7 days if not specified
    except ValueError:
        return jsonify({"error": "Invalid 'days' parameter. Please provide an integer."}), 400

    if days <= 0:
        return jsonify({"error": "Number of days must be a positive integer."}), 400

    try:
        # Create a future dataframe for the specified number of days
        future = model.make_future_dataframe(periods=days)

        # Make predictions
        forecast = model.predict(future)

        # Extract relevant columns and convert to a list of dictionaries
        # .tail(days) ensures we only return the newly predicted future values
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days).to_dict(orient='records')

        # Convert datetime objects to string for JSON serialization
        for item in result:
            item['ds'] = item['ds'].strftime('%Y-%m-%d')

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode (for development)
    # In production, use a production-ready WSGI server like Gunicorn or uWSGI
    app.run(debug=True)
