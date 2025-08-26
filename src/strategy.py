import pandas as pd
import pandas_ta as ta

class EMAStrategy:
    """
    Implements the EMA crossover trading strategy with multiple filters and confirmations.
    """
    
    def __init__(self, 
                 ema_short: int = 9, 
                 ema_long: int = 20):
        """
        Initialize the EMA strategy with parameters.
        
        Args:
            ema_short (int): Short-term EMA period
            ema_long (int): Long-term EMA period
            risk_per_trade (float): Maximum risk per trade as percentage of portfolio
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
        """
        self.ema_short = ema_short
        self.ema_long = ema_long

    def get_entry_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify entry conditions based on EMA crossover and desired indicators.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with entry signals
        """
        data = self.generate_signals(data)
        data = self.ADX_filter(data, adx_threshold=25)
        
        return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on EMA crossover with multiple filters.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with trading signals
        """
        df = data.copy()

        # Generate base crossover signals
        df['signal'] = 0
        df.loc[(df[f'EMA_{self.ema_short}'] > df[f'EMA_{self.ema_long}']) & 
               (df[f'EMA_{self.ema_short}'].shift(1) <= df[f'EMA_{self.ema_long}'].shift(1)), 'signal'] = 1
        df.loc[(df[f'EMA_{self.ema_short}'] < df[f'EMA_{self.ema_long}']) & 
               (df[f'EMA_{self.ema_short}'].shift(1) >= df[f'EMA_{self.ema_long}'].shift(1)), 'signal'] = -1
        
        return df
    
    def ADX_filter(self, data: pd.DataFrame, adx_threshold: float = 25.0) -> pd.DataFrame:
        """
        Apply ADX filter to the signals.
        
        Args:
            data (pd.DataFrame): Market data with ADX indicator
            adx_threshold (float): Minimum ADX value to confirm trend
            
        Returns:
            pd.DataFrame: Data with filtered signals
        """
        df = data.copy()
        
        # Filter signals
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        df['ADX'] = df['ADX'].fillna(0)
        df.loc[(df['signal'] != 0) & (df['ADX'] < adx_threshold), 'signal'] = 0
        
        return df