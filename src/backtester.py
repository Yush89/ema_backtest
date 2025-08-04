from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .strategy import SMAStrategy
from .data_handler import DataHandler

class Backtester:
    """
    Main backtesting engine for evaluating trading strategies.
    """
    
    def __init__(self,
                 strategy: SMAStrategy,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0001):
        """
        Initialize the backtester.
        
        Args:
            strategy (SMAStrategy): Trading strategy instance
            initial_capital (float): Initial capital for backtesting
            commission (float): Commission rate per trade
            slippage (float): Slippage rate per trade
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.portfolio = None
        self.trades = []
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest.
        
        Args:
            data (pd.DataFrame): Market data with signals
            
        Returns:
            pd.DataFrame: Portfolio performance data
        """
        # Initialize portfolio tracking
        self.portfolio = self._initialize_portfolio(data.index)
        positions = pd.DataFrame(0, index=data.index, columns=['shares'])
        
        # Run through each bar
        for i in range(1, len(data)):
            current_bar = data.iloc[i]
            prev_bar = data.iloc[i-1]
            
            # Check for trade signals
            if current_bar['position'] != 0:
                self._execute_trade(
                    current_bar,
                    positions,
                    i,
                    current_bar['position']
                )
            else:
                # Update portfolio value for hold positions
                positions.iloc[i] = positions.iloc[i-1]
                self.portfolio.loc[current_bar.name, 'position_value'] = (
                    positions.iloc[i]['shares'] * current_bar['close']
                )
                self.portfolio.loc[current_bar.name, 'cash'] = (
                    self.portfolio.iloc[i-1]['cash']
                )
            
            # Update total portfolio value
            self.portfolio.loc[current_bar.name, 'total_value'] = (
                self.portfolio.loc[current_bar.name, 'position_value'] +
                self.portfolio.loc[current_bar.name, 'cash']
            )
        
        return self.portfolio
    
    def _initialize_portfolio(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Initialize the portfolio DataFrame.
        
        Args:
            index (pd.DatetimeIndex): Time index for the portfolio
            
        Returns:
            pd.DataFrame: Initialized portfolio DataFrame
        """
        portfolio = pd.DataFrame(index=index)
        portfolio['position_value'] = 0.0
        portfolio['cash'] = self.initial_capital
        portfolio['total_value'] = self.initial_capital
        return portfolio
    
    def _execute_trade(self,
                      bar: pd.Series,
                      positions: pd.DataFrame,
                      index: int,
                      signal: float) -> None:
        """
        Execute a trade and update the portfolio.
        
        Args:
            bar (pd.Series): Current price bar
            positions (pd.DataFrame): Current positions
            index (int): Current index
            signal (float): Trade signal (1 for buy, -1 for sell)
        """
        current_price = bar['close']
        
        if signal > 0:  # Buy signal
            # Calculate position size
            price_with_slippage = current_price * (1 + self.slippage)
            position_size = self.strategy.calculate_position_size(
                price_with_slippage,
                self.portfolio.iloc[index-1]['total_value']
            )
            
            # Calculate costs
            trade_cost = (position_size * price_with_slippage * 
                         (1 + self.commission))
            
            # Update positions and portfolio
            positions.iloc[index] = position_size
            self.portfolio.iloc[index, self.portfolio.columns.get_loc('position_value')] = (
                position_size * current_price
            )
            self.portfolio.iloc[index, self.portfolio.columns.get_loc('cash')] = (
                self.portfolio.iloc[index-1]['cash'] - trade_cost
            )
            
            # Record trade
            self.trades.append({
                'timestamp': bar.name,
                'type': 'buy',
                'shares': position_size,
                'price': price_with_slippage,
                'cost': trade_cost
            })
            
        elif signal < 0:  # Sell signal
            # Calculate position size and value
            shares_to_sell = positions.iloc[index-1]['shares']
            price_with_slippage = current_price * (1 - self.slippage)
            trade_value = shares_to_sell * price_with_slippage
            trade_cost = trade_value * self.commission
            
            # Update positions and portfolio
            positions.iloc[index] = 0
            self.portfolio.iloc[index, self.portfolio.columns.get_loc('position_value')] = 0
            self.portfolio.iloc[index, self.portfolio.columns.get_loc('cash')] = (
                self.portfolio.iloc[index-1]['cash'] + trade_value - trade_cost
            )
            
            # Record trade
            self.trades.append({
                'timestamp': bar.name,
                'type': 'sell',
                'shares': shares_to_sell,
                'price': price_with_slippage,
                'cost': trade_cost
            })
