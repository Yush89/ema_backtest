import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

class PerformanceAnalyzer:
    """
    Analyzes and visualizes trading strategy performance.
    """
    
    def __init__(self, portfolio: pd.DataFrame, trades: List[Dict], data: pd.DataFrame):
        """
        Initialize the performance analyzer.
        
        Args:
            portfolio (pd.DataFrame): Portfolio performance data
            trades (List[Dict]): List of executed trades
            data (pd.DataFrame): Original market data with signals
        """
        self.portfolio = portfolio
        self.trades = pd.DataFrame(trades)
        self.data = data
        self.metrics = {}
        
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            Dict: Dictionary of performance metrics
        """
        # Calculate returns
        self.portfolio['returns'] = self.portfolio['total_value'].pct_change()
        
        # Total Return
        total_return = (
            (self.portfolio['total_value'].iloc[-1] - 
             self.portfolio['total_value'].iloc[0]) /
            self.portfolio['total_value'].iloc[0]
        )
        
        # Calculate Sharpe Ratio
        excess_returns = self.portfolio['returns'] - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate Maximum Drawdown
        cumulative_returns = (1 + self.portfolio['returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate Win Rate
        winning_trades = self.trades[self.trades['type'] == 'sell']
        win_rate = len(winning_trades[winning_trades['price'] > winning_trades['price'].shift(1)]) / len(winning_trades)
        
        self.metrics = {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Number of Trades': len(self.trades) // 2  # Divide by 2 as each round trip is 2 trades
        }
        
        return self.metrics
    
    def plot_results(self) -> None:
        """
        Create visualizations of the backtest results.
        """
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Price and SMAs
        ax1.plot(self.data.index, self.data['close'], label='Price')
        ax1.plot(self.data.index, self.data['SMA_9'], label='SMA 9')
        ax1.plot(self.data.index, self.data['SMA_20'], label='SMA 20')
        ax1.set_title('Price and Moving Averages')
        ax1.legend()
        
        # Plot 2: Portfolio Value
        ax2.plot(self.portfolio.index, self.portfolio['total_value'])
        ax2.set_title('Portfolio Value')
        
        # Plot 3: Drawdown
        returns = self.portfolio['total_value'].pct_change()
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        
        plt.tight_layout()
        plt.show()
    
    def generate_trade_report(self) -> pd.DataFrame:
        """
        Generate a detailed trade report.
        
        Returns:
            pd.DataFrame: Trade report with performance metrics
        """
        # Convert trades list to DataFrame if it's not already
        if isinstance(self.trades, list):
            trades_df = pd.DataFrame(self.trades)
        else:
            trades_df = self.trades.copy()
        
        # Calculate trade-specific metrics
        trades_df['profit'] = trades_df['price'] * trades_df['shares']
        trades_df['profit'] = trades_df['profit'].diff()
        
        # Remove buy trades (they don't have profit information)
        trades_df = trades_df[trades_df['type'] == 'sell'].copy()
        
        # Calculate additional metrics
        trades_df['return'] = trades_df['profit'] / trades_df['cost']
        
        return trades_df
