import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("Connecting to Yahoo Finance...")

try:
    # Use 'download' directly - it's the most stable method in the new version
    # We'll use AAPL as it's the most reliable ticker for testing
    data = yf.download('AAPL', period='5y')
    
    if data.empty:
        raise ValueError("Data download failed. Check your internet connection.")

    print("✅ Data Downloaded! Preparing training sequences...")
    
    # We only need the 'Close' prices
    training_set = data['Close'].values.reshape(-1,1)

    # 1. Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(training_set)

    # 2. Create the 60-day window sequences
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 3. Build the LSTM Architecture
    print("Training the AI (This will take a few seconds)...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # We use 1 epoch just to get the file created quickly so we can deploy!
    model.fit(X_train, y_train, epochs=1, batch_size=32)

    # 4. Save the "Brain" of your project
    model.save("stock_lstm_model.h5")
    print("\n🚀 SUCCESS! 'stock_lstm_model.h5' created in your folder.")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")