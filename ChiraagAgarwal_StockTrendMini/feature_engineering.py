import pandas as pd
import os


def create_features(file_path="data/AAPL_data.csv", save_folder="data"):

    print("\nCreating Features...\n")

    df = pd.read_csv(file_path)

    # Convert Date column 
    df["Date"] = pd.to_datetime(df["Date"])
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date")

    
    # Daily Return
    df["Daily_Return"] = df["Close"].pct_change()

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    df["MA_50"] = df["Close"].rolling(window=50).mean()

   
    df["Volatility"] = df["Close"].rolling(window=10).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

   
    df.dropna(inplace=True)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    feature_file = os.path.join(save_folder, "AAPL_features.csv")
    df.to_csv(feature_file, index=False)

    print("✅ Features Created Successfully!")
    print(f"📁 Saved at: {feature_file}")
    print(f"📊 Total Rows After Cleaning: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    create_features()
    