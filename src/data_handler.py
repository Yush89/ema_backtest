import pandas as pd
import os
from typing import Optional
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class DataHandler:
    """
    Handles data loading, cleaning, and preprocessing for the backtesting system.
    """
    
    def __init__(self, api_key: str, secret_key: str, symbol: str):
        """
        Initialize DataHandler with Alpaca API credentials.
        
        Args:
            api_key (str): Alpaca API key
            secret_key (str): Alpaca API secret key
        """
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.data = None
        self.api_calls = 0
        self.last_api_call = None
        self.rate_limit = 200  # Alpaca's rate limit per minute
        self.symbol = symbol
        self.csv_path = os.path.join(base_dir, f'{symbol.lower()}_data.csv')
        
    def _check_rate_limit(self):
        """
        Check if we're within API rate limits.
        Raises:
            Exception: If rate limit would be exceeded
        """
        current_time = datetime.now()
        
        # Reset counter if it's been more than a minute
        if self.last_api_call and (current_time - self.last_api_call) > timedelta(minutes=1):
            self.api_calls = 0
            
        # Check if we're about to exceed rate limit
        if self.api_calls >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_api_call).seconds
            if wait_time > 0:
                raise Exception(f"Rate limit reached. Please wait {wait_time} seconds before making another request.")
    
    def _load_from_csv(self) -> Optional[pd.DataFrame]:
        """
        Try to load data from CSV file. Checks if file exists and has valid content.
        Returns None if file doesn't exist, is empty, or has invalid data.
        """
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            try:
                df = pd.read_csv(self.csv_path)
                if len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    # Validate the loaded data
                    try:
                        self._validate_data(df)
                        print(f"Successfully loaded data from {self.csv_path}")
                        return df
                    except ValueError as e:
                        print(f"Data validation failed: {e}")
                        return None
                else:
                    print("CSV file is empty")
                    return None
            except Exception as e:
                print(f"Error loading CSV: {e}")
                return None
        return None
    
    def _save_to_csv(self, df: pd.DataFrame) -> None:
        """
        Save data to CSV file.
        """
        # Reset index to make timestamp a column
        df_to_save = df.reset_index()
        df_to_save.to_csv(self.csv_path, index=False)
        
    def fetch_data(self, 
                    symbol: str,
                    timeframe: TimeFrame = TimeFrame.Hour,
                    start: datetime = datetime(2022, 1, 1),
                    end: datetime = datetime(2025, 8, 1)) -> pd.DataFrame:
        """
        Fetch market data from Alpaca API or load from CSV if available.
        
        Args:
            symbol (str): Stock symbol to fetch
            timeframe (TimeFrame): Data timeframe (Hour, Day, etc.)
            start (datetime): Start date for historical data
            end (datetime): End date for historical data
            
        Returns:
            pd.DataFrame: Cleaned and preprocessed market data
        """
        # Try to load from CSV first
        df = self._load_from_csv()
        if df is not None:
            print(f"Data loaded from {self.csv_path}")
            self.data = df
            return self.data
            
        # If no CSV or invalid data, fetch from API
        print("Fetching data from Alpaca API...")
        self._check_rate_limit()
        
        # Request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end
        )
        
        # Fetch data and update rate limit tracking
        bars = self.client.get_stock_bars(request_params)
        self.api_calls += 1
        self.last_api_call = datetime.now()
        
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
        
        # Save to CSV
        self._save_to_csv(df)
        print(f"Data saved to {self.csv_path}")
        
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
            
    def add_technical_indicators(self, ema_short: int, ema_long: int) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            ema_short (int): Short-term EMA period
            ema_long (int): Long-term EMA period
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        df = self.data.copy()
        
        # Calculate EMAs with generic names
        df['EMA_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
        
        return df
