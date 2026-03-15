import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from curl_cffi import requests as curl_requests # The secret fix

app = Flask(__name__)

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
        # Create a session that looks like Google Chrome
        session = curl_requests.Session(impersonate="chrome")
        
        # Download data using the browser-like session
        df = yf.download(ticker, period='2y', interval='1d', progress=False, session=session)
        
        if df.empty:
            return render_template('index.html', error=f"Ticker '{ticker}' not found or rate limited.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        data = df[['Close']].values
        if len(data) < 60:
            return render_template('index.html', error="Not enough historical data.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        prediction_scaled = model.predict(last_60_days, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        result = round(float(prediction[0][0]), 2)
        last_close = round(float(data[-1][0]), 2)

        return render_template('index.html', ticker=ticker, prediction=result, last_close=last_close)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Yahoo Finance is temporarily limiting requests. Try again in 1 minute.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)