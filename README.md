# 📈 Stock Sentiment Predictor

A machine learning web app that predicts next-day stock price movement by combining technical price indicators with NLP-based news sentiment analysis.

## 🔗 Live App
https://stock-sentiment-predictor-9ckzpysbshhlpbekcszkw7.streamlit.app/

## 📌 What it does
- Downloads real stock data for AAPL, TSLA and MSFT using yfinance
- Applies VADER sentiment analysis on financial news headlines
- Trains and compares Random Forest and XGBoost classifiers
- Shows interactive charts for stock price, sentiment and feature importance
- Predicts whether the stock will go UP or DOWN tomorrow

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- yfinance, VADER Sentiment
- Scikit-learn, XGBoost
- Streamlit, Matplotlib, Seaborn

## 📊 Key Finding
Sentiment ranked as the #4 most important feature, proving that news sentiment adds measurable predictive value beyond price data alone.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
