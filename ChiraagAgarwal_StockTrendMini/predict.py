import pandas as pd
from sklearn.linear_model import LogisticRegression


def predict_next_day(file_path="data/AAPL_features.csv"):

    print("\nLoading dataset...\n")

    # ===
    df = pd.read_csv(file_path)

    features = [
        "Daily_Return",
        "MA_10",
        "MA_50",
        "Volatility"
    ]

    X = df[features]
    y = df["Target"]

    model = LogisticRegression()
    model.fit(X, y)

    latest_data = X.iloc[-1:]

    # =========================
    prediction = model.predict(latest_data)[0]

    # ============================
    if prediction == 1:
        print("📈 Prediction: Stock price will likely go UP tomorrow.")
    else:
        print("📉 Prediction: Stock price will likely go DOWN tomorrow.")

    return prediction


if __name__ == "__main__":
    predict_next_day()