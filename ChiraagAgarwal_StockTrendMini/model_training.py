import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model(file_path="data/AAPL_features.csv"):

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

    model = LogisticRegression()

    model.fit(X_train, y_train)

    print("✅ Model Training Completed")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n📊 Model Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


if __name__ == "__main__":
    train_model()