import os

# --- STEP 1: FORCE MEMORY SAVINGS (Must be at the very top) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# Limit TensorFlow to 1 CPU core to prevent RAM spikes
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model once at startup to save time
MODEL_PATH = 'stock_model.h5'
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', '').upper().strip()
    if not ticker:
        return render_template('index.html', error="Please enter a ticker symbol.")

    try:
        # --- STEP 2: STEALTH DATA FETCHING ---
        # We fetch slightly more data (2 years) to ensure we have enough for the 60-day window
        df = yf.download(ticker, period='2y', interval='1d', progress=False)
        
        if df.empty:
            return render_template('index.html', error=f"Ticker '{ticker}' not found or no data available.")

        # Handle multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Use 'Close' prices for prediction
        data = df[['Close']].values
        if len(data) < 60:
            return render_template('index.html', error="Not enough historical data (need at least 60 days).")

        # --- STEP 3: PREDICTION LOGIC ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Get the last 60 days to predict tomorrow
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        
        # Run prediction
        prediction_scaled = model.predict(last_60_days, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        result = round(float(prediction[0][0]), 2)
        last_close = round(float(data[-1][0]), 2)

        return render_template('index.html', 
                               ticker=ticker, 
                               prediction=result, 
                               last_close=last_close)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    # Use port from environment (important for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)