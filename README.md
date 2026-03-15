# 📈 StockPulse AI
A deep-learning web application that predicts stock prices using Long Short-Term Memory (LSTM) networks.

## 🚀 Live Demo
[**View Live App**](https://stockpulse-ai-nssa.onrender.com)
*(Note: Uses Alpha Vantage Free Tier. If the "API Limit" message appears, please wait 60 seconds or try again tomorrow as the daily limit is 25 calls.)*

## 📸 Project Snapshot
![StockPulse Interface]("C:\Users\HP\Desktop\StockPulse_ai.png")

## 🛠️ Key Features
* **Secure Architecture:** Uses environment variables for API key protection (zero hard-coded secrets).
* **Robust Data Pipeline:** Integrated with Alpha Vantage API for real-time historical financial data.
* **Machine Learning:** LSTM model trained on 10+ years of historical stock data.
* **Cloud Deployment:** Optimized for Render with Gunicorn-threaded workers and lazy-loading for heavy AI models.

## 💻 Tech Stack
* **Frontend:** HTML5, CSS3 (Dark Mode UI)
* **Backend:** Flask (Python)
* **AI/ML:** TensorFlow, Keras, Scikit-Learn
* **Deployment:** Render, Gunicorn, Git