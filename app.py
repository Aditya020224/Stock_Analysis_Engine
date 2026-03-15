import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress scary TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Reduce memory overhead
import yfinance as yf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# Add this line below your imports:
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# Load the model you just created
MODEL_PATH = "stock_lstm_model.h5"
model = load_model(MODEL_PATH)
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form.get('ticker').upper()
        # Fetching fresh data for the user's ticker
        df = yf.download(ticker, period="1y")
        
        if df.empty:
            return render_template('index.html', error="Ticker not found!")
        
        # Prepare the data
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        
        # Take the last 60 days to predict tomorrow
        input_data = scaled_data[-60:].reshape(1, 60, 1)
        
        # AI Prediction
        prediction_scaled = model.predict(input_data)
        final_price = scaler.inverse_transform(prediction_scaled)
        
        return render_template('index.html', 
                               prediction=round(float(final_price[0][0]), 2), 
                               ticker=ticker)
    except Exception as e:
        return render_template('index.html', error="Error processing request. Try a different ticker.")

if __name__ == '__main__':
    # Use environment port for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)