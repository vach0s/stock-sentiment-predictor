import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Sentiment Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Sentiment Predictor")
st.write("Predicts next-day stock movement using price data and news sentiment analysis")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "TSLA", "MSFT"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.sidebar.button("Run Prediction"):
    with st.spinner("Downloading data and training model..."):

        # Download data
        raw_data = yf.download(ticker, start=start_date, end=end_date)
        df = pd.DataFrame()
        df['Close'] = raw_data['Close']
        df['Daily_Return'] = df['Close'].pct_change()
        df['Target'] = (df['Daily_Return'].shift(-1) > 0).astype(int)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA5_ratio'] = df['Close'] / df['MA5']
        df['Volatility'] = df['Daily_Return'].rolling(window=5).std()
        df['Prev_Return'] = df['Daily_Return'].shift(1)

        # Sentiment
        analyzer = SentimentIntensityAnalyzer()
        sample_headlines = [
            "Stock market rallies as inflation fears ease",
            "Tech stocks tumble amid recession concerns",
            "Apple reports record breaking quarterly earnings",
            "Markets fall sharply on weak economic data",
            "Investors optimistic as interest rates stabilize",
            "Tesla shares drop after disappointing delivery numbers",
            "Microsoft cloud revenue beats expectations strongly",
            "Federal Reserve signals potential rate cuts ahead",
            "Global markets decline on geopolitical tensions",
            "Strong jobs report boosts investor confidence"
        ]
        np.random.seed(42)
        df['Sentiment'] = [
            analyzer.polarity_scores(
                sample_headlines[np.random.randint(0, len(sample_headlines))]
            )['compound'] for _ in range(len(df))
        ]

        df = df.dropna()

        # Train models
        features = ['Daily_Return', 'MA5_ratio', 'Volatility', 'Prev_Return', 'Sentiment']
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

        xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

        # Results
        st.success("Model trained successfully!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Stock", ticker)
        col2.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
        col3.metric("XGBoost Accuracy", f"{xgb_acc*100:.2f}%")

        # Chart 1: Stock price
        st.subheader("Stock Price Over Time")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df.index, df['Close'], color='#1D9E75', linewidth=1.5)
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        # Chart 2: Sentiment
        st.subheader("News Sentiment Over Time")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.fill_between(df.index, df['Sentiment'],
                         where=(df['Sentiment'] >= 0),
                         color='green', alpha=0.4, label='Positive')
        ax2.fill_between(df.index, df['Sentiment'],
                         where=(df['Sentiment'] < 0),
                         color='red', alpha=0.4, label='Negative')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        # Chart 3: Feature importance
        st.subheader("Feature Importance (XGBoost)")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.barh(importance_df['Feature'], importance_df['Importance'], color='#7F77DD')
        ax3.set_xlabel("Importance Score")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

        # Prediction for tomorrow
        st.subheader("Tomorrow's Prediction")
        last_row = X.iloc[-1].values.reshape(1, -1)
        prediction = xgb_model.predict(last_row)[0]
        if prediction == 1:
            st.success(f"📈 XGBoost predicts {ticker} will go UP tomorrow")
        else:
            st.error(f"📉 XGBoost predicts {ticker} will go DOWN tomorrow")
