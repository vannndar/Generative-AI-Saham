import yfinance as yf
import pandas as pd
import os

def download_data(stock):
    # Define the ticker symbol
    ticker_symbol = stock

    # Create a Ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Fetch historical market data
    historical_data = ticker.history(period="5y")  # Data for the last 5 years
    # Ensure "Date" column is datetime without timezone
    historical_data.index = pd.to_datetime(historical_data.index).tz_localize(None)

    # Add a 'date_str' column
    historical_data['date_str'] = ''

    # Loop to print the date in "YYYY-MM-DD" format and "Close" value
    for i in range(len(historical_data)):
        date_str = historical_data.index[i].strftime('%Y-%m-%d')
        close_value = historical_data['Close'][i]
        # Add new 'date_str' column as datetime with the date value
        historical_data['date_str'][i] = pd.to_datetime(date_str)
        print(f"{date_str}: {close_value}")
    
    # Corrected file path creation using split(".")
    location = os.path.join('data', f'{stock.split(".")[0]}_jk.csv')
    # create a folder if it does not exist
    locationmethod = os.path.join('method', f'{stock.split(".")[0]}')
    print(locationmethod)
    os.makedirs(locationmethod, exist_ok=True)  # This ensures that intermediate directories are created if they don't exist
    
    
    # Save to CSV file
    historical_data.to_csv(location, index=False)
