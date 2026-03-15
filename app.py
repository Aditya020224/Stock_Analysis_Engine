import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Professional memory optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# Fetch API Key from Render environment (Secure move!)
API_KEY = os.environ.get('STOCK_API_KEY')
MODEL_PATH = 'stock_model.h5'

# Load model once at startup
model = load_model(MODEL_PATH)

def fetch_alpha_vantage_data(symbol):
    """Professional data fetching with error handling"""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'compact'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Handle API errors or rate limits (25 calls/day free tier)
    if "Error Message" in data:
        return None, f"Ticker '{symbol}' not found."
    if "Note" in data:
        return None, "API Rate Limit reached. Please wait 1 minute."
    if "Time Series (Daily)" not in data:
        return None, "Technical error fetching data."
        
    # Convert JSON to a clean Pandas DataFrame
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Alpha Vantage column names are '4. close', '1. open', etc.
    # We rename to 'Close' to keep our logic consistent
    df = df.rename(columns={'4. close': 'Close'})
    return df[['Close']], None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', '').upper().strip()
    if not ticker:
        return render_template('index.html', error="Please enter a ticker symbol.")

    # 1. Fetching
    df, error = fetch_alpha_vantage_data(ticker)
    if error:
        return render_template('index.html', error=error)

    try:
        # 2. Pre-processing (Scale the last 60 days)
        data_values = df.values
        if len(data_values) < 60:
            return render_template('index.html', error="Not enough historical data (need 60 days).")
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)

        # 3. Model Prediction
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        prediction_scaled = model.predict(last_60_days, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        # 4. Final Results
        result = round(float(prediction[0][0]), 2)
        last_close = round(float(data_values[-1][0]), 2)

        return render_template('index.html', 
                               ticker=ticker, 
                               prediction=result, 
                               last_close=last_close)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html', error="AI processing failed. Check logs.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)