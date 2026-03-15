import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 1. Compatibility Patch for Keras/TensorFlow versions
import keras
if hasattr(keras, 'config'):
    keras.config.enable_unsafe_deserialization()

# 2. Memory & Speed Optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# Global variables
API_KEY = os.environ.get('STOCK_API_KEY')
MODEL_PATH = 'stock_lstm_model.h5'
model = None  # We leave this empty for "Lazy Loading"

def fetch_alpha_vantage_data(symbol):
    """Fetches data from official Alpha Vantage API"""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'compact'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if "Time Series (Daily)" not in data:
            return None, "API Limit reached or Ticker invalid. Try in 1 minute."
        
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().rename(columns={'4. close': 'Close'})
        return df[['Close']], None
    except Exception:
        return None, "Network error. Please try again."

@app.route('/')
def index():
    global model
    # LAZY LOADING: This prevents the 'Port Timeout' error on Render
    if model is None:
        try:
            model = load_model(MODEL_PATH, compile=False)
        except Exception as e:
            print(f"Model load error: {e}")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    ticker = request.form.get('ticker', '').upper().strip()
    
    # Ensure model is loaded
    if model is None:
        model = load_model(MODEL_PATH, compile=False)

    # 1. Get Data
    df, error = fetch_alpha_vantage_data(ticker)
    if error:
        return render_template('index.html', error=error)

    try:
        # 2. Process Data
        data_values = df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)
        
        # 3. Predict
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        prediction_scaled = model.predict(last_60_days, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        res = round(float(prediction[0][0]), 2)
        last_c = round(float(data_values[-1][0]), 2)

        return render_template('index.html', ticker=ticker, prediction=res, last_close=last_c)
    except Exception:
        return render_template('index.html', error="AI Prediction failed. Check ticker.")

if __name__ == '__main__':
    # Render provides the PORT variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)