import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title("📈 StockTrendMini AI Predictor")

ticker = st.text_input("Enter Stock Ticker", "AAPL")


def download_data(ticker):
    data = yf.download(ticker, period="1y")
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.reset_index(inplace=True)
    return data


def create_features(df):

    # Feature Engineering
    df["Daily_Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Daily_Return"].rolling(10).std()

    # Target Variable
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove NaNs created by rolling / shift
    df.dropna(inplace=True)

    return df


if st.button("Run Prediction"):

    st.write("Downloading latest market data...")

    df = download_data(ticker)
    df = create_features(df)

    features = ["Daily_Return", "MA_10", "MA_50", "Volatility"]

    # Prepare training data
    X = df[features]
    y = df["Target"]

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Predict using latest data
    latest = X.iloc[-1:]
    prediction = model.predict(latest)[0]

    if prediction == 1:
        st.success("📈 Prediction: Price likely to go UP tomorrow")
    else:
        st.error("📉 Prediction: Price likely to go DOWN tomorrow")

    st.subheader("Recent Closing Prices")

    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title(f"{ticker} Closing Price Trend")

    st.pyplot(fig)

    st.subheader("Latest Features Used For Prediction")
    st.write(latest)