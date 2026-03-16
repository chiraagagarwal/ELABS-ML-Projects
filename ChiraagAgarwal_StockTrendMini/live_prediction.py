import yfinance as yf
import pandas as pd
import joblib


def get_latest_data(ticker="AAPL"):
    
    print("\nDownloading latest stock data...\n")

    data = yf.download(ticker, period="6mo")

    data = data[["Open", "High", "Low", "Close", "Volume"]]

    data.reset_index(inplace=True)

    return data


def create_features(df):

    df["Daily_Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].rolling(10).std()

    df.dropna(inplace=True)

    return df


def predict_live():

    print("Loading trained model...\n")

    model = joblib.load("stock_model.pkl")

    df = get_latest_data()

    df = create_features(df)

    features = [
        "Daily_Return",
        "MA_10",
        "MA_50",
        "Volatility"
    ]

    latest = df[features].iloc[-1:]

    prediction = model.predict(latest)[0]

    print("Latest Market Data:")
    print(latest)

    if prediction == 1:
        print("\n📈 Prediction: Stock likely to go UP tomorrow")
    else:
        print("\n📉 Prediction: Stock likely to go DOWN tomorrow")

    # Feature Importance
    importance = model.feature_importances_

    print("\nFeature Importance:")

    for name, score in zip(features, importance):
        print(f"{name} : {score:.3f}")


if __name__ == "__main__":
    predict_live()