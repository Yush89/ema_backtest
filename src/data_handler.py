import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class DataHandler:
    """
    Handles data loading, cleaning, and preprocessing for the backtesting system.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Initialize DataHandler with Alpaca API credentials.
        
        Args:
            api_key (str): Alpaca API key
            secret_key (str): Alpaca API secret key
        """
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.data = None
        
    def fetch_data(self, 
                    symbol: str,
                    timeframe: TimeFrame = TimeFrame.Hour,
                    start: datetime = datetime(2022, 1, 1),
                    end: datetime = datetime(2025, 8, 1)) -> pd.DataFrame:
        """
        Fetch market data from Alpaca API.
        
        Args:
            symbol (str): Stock symbol to fetch
            timeframe (TimeFrame): Data timeframe (Hour, Day, etc.)
            start (datetime): Start date for historical data
            end (datetime): End date for historical data
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed market data
        """
        # Request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end
        )
        
        # Fetch data
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        
        # Drop the symbol column and set timestamp as index
        if 'symbol' in df.columns:
            df = df.drop('symbol', axis=1)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Validate data
        self._validate_data(df)
        
        # Store processed data
        self.data = df
        
        return self.data
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the data for common issues.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
        
        Raises:
            ValueError: If data validation fails
        """
        # Check for missing values
        if df.isnull().any().any():
            raise ValueError("Dataset contains missing values")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            raise ValueError("Dataset contains duplicate timestamps")
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
    def add_technical_indicators(self, sma_short: int = 9, sma_long: int = 20) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            sma_short (int): Short-term SMA period
            sma_long (int): Long-term SMA period
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        df = self.data.copy()
        
        # Calculate SMAs
        df[f'SMA_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
        df[f'SMA_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
        
        # Calculate crossover signals
        df['signal'] = 0
        df.loc[df[f'SMA_{sma_short}'] > df[f'SMA_{sma_long}'], 'signal'] = 1
        df.loc[df[f'SMA_{sma_short}'] < df[f'SMA_{sma_long}'], 'signal'] = -1
        
        return df
