from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

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
    