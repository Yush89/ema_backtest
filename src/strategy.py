from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class SMAStrategy:
    """
    Implements the SMA crossover trading strategy.
    """
    
    def __init__(self, 
                 sma_short: int = 9, 
                 sma_long: int = 20,
                 risk_per_trade: float = 0.02,
                 stop_loss_pct: float = 0.02):
        """
        Initialize the SMA strategy with parameters.
        
        Args:
            sma_short (int): Short-term SMA period
            sma_long (int): Long-term SMA period
            risk_per_trade (float): Maximum risk per trade as percentage of portfolio
            stop_loss_pct (float): Stop loss percentage
        """
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on SMA crossover.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with trading signals
        """
        df = data.copy()
        
        # Generate position entry signals
        df['position'] = df['signal'].diff()
        
        return df
    
    def calculate_position_size(self, 
                              price: float, 
                              portfolio_value: float) -> float:
        """
        Calculate the position size based on risk management rules.
        
        Args:
            price (float): Current asset price
            portfolio_value (float): Current portfolio value
            
        Returns:
            float: Number of shares to trade
        """
        # Calculate maximum loss allowed per trade
        max_risk_amount = portfolio_value * self.risk_per_trade
        
        # Calculate stop loss price
        stop_loss_amount = price * self.stop_loss_pct
        
        # Calculate position size
        position_size = max_risk_amount / stop_loss_amount
        
        return np.floor(position_size)
    
    def validate_signal(self, row: pd.Series) -> bool:
        """
        Validate if a signal should be taken based on additional criteria.
        
        Args:
            row (pd.Series): Current market data row
            
        Returns:
            bool: Whether the signal should be taken
        """
        # Add additional validation criteria here
        # For now, return True for all signals
        return True
