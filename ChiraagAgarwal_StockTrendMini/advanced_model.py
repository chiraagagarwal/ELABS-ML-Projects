import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_advanced_model(file_path="data/AAPL_features.csv"):

    print("\nLoading dataset...\n")
    df = pd.read_csv(file_path)

    features = [
        "Daily_Return",
        "MA_10",
        "MA_50",
        "Volatility"
    ]

    X = df[features]
    y = df["Target"]

    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✅ Random Forest Model Trained")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n📊 Model Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


    joblib.dump(model, "stock_model.pkl")

    print("\n💾 Model saved as stock_model.pkl")

  
    plt.figure(figsize=(10,5))
    plt.plot(df["Close"])
    plt.title("Stock Closing Price Trend")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

    return model


if __name__ == "__main__":
    train_advanced_model()