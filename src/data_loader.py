import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def prepare_task1_data():
    # p(t): Stock, s(t): Sensex, d(t): USD-INR
    tickers = ["TCS.NS", "^BSESN", "INR=X"]
    
    print("Fetching multivariate signal data...")
    # Download data with safety flags to prevent WSL hanging
    raw_data = yf.download(tickers, start="2022-01-01", end="2025-01-01", progress=False, threads=False)['Close']
    
    # Clean and Align (Task 1)
    df = raw_data.ffill().dropna()
    
    # Normalize (Task 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    
    processed_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    
    # Create data dir if not exists
    os.makedirs('data', exist_ok=True)
    processed_df.to_csv('data/processed_data.csv')
    print("Task 1 Complete. Aligned data saved to data/processed_data.csv")
    return processed_df

if __name__ == "__main__":
    prepare_task1_data()