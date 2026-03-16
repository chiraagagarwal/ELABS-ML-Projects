import yfinance as yf
import pandas as pd
import os


def download_stock_data(ticker="AAPL",
                        start_date="2015-01-01",
                        end_date="2024-01-01",
                        save_path="data"):
    

    print(f"Downloading data for {ticker}...")

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print("No data found. Check ticker symbol or internet connection.")
        return None


    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data[required_columns]

    
    data.reset_index(inplace=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #
    file_name = f"{save_path}/{ticker}_data.csv"
    data.to_csv(file_name, index=False)

    print(f"Data saved successfully to {file_name}")
    print("First 5 rows:")
    print(data.head())

    return data


if __name__ == "__main__":
    download_stock_data(
        ticker="AAPL",
        start_date="2015-01-01",
        end_date="2024-01-01"
    )