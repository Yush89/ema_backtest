#Endpoint : https://paper-api.alpaca.markets/v2
#API_KEY : PKOUNE8KZRCZ7SF6MK54
#Secret : 36ZtbKBFNDs8b1qFnB9tStsNhRUHIDjbdOkRcYgO


from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pandas as pd

# Replace with your keys
API_KEY = "PKOUNE8KZRCZ7SF6MK54"
SECRET_KEY = "36ZtbKBFNDs8b1qFnB9tStsNhRUHIDjbdOkRcYgO"

# Create the client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Request historical daily bars for AAPL
request_params = StockBarsRequest(
    symbol_or_symbols=["AMD"],
    timeframe=TimeFrame.Day,
    start=datetime(2022, 1, 1),
    end=datetime(2025, 8, 1)
)

bars = client.get_stock_bars(request_params)
df = bars.df

# Save to CSV if you want
df.to_csv("/Users/yush/Documents/Side_hustles/Day_Trading/Data_analysis/sma-backtest/amd_data.csv")

print(df.head())