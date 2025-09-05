import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceAnalyzer:
    """
    Analyzes and visualizes trading strategy performance.
    """
    
    def __init__(self, portfolio: pd.DataFrame, trades: pd.DataFrame, data: pd.DataFrame):
        """
        Initialize the performance analyzer.
        
        Args:
            portfolio (pd.DataFrame): Portfolio performance data
            trades (pd.DataFrame): Executed trades data
            data (pd.DataFrame): Original market data with signals
        """
        self.portfolio = portfolio
        self.trades = trades
        self.data = data
        self.metrics = {}
    
    def analyze_trades(self):
        """
        Analyze the trade log and compute performance metrics.
        Returns:
            dict: Dictionary of performance metrics
        """
        if self.trades is None or self.trades.empty:
            raise ValueError("No trades to analyze. Run log_trades first.")

        trades_df = self.trades.copy()

        # Ensure pct_gain_loss column is numeric and drop NaNs
        trades_df['pct_gain_loss'] = pd.to_numeric(trades_df['pct_gain_loss'], errors='coerce')
        trades_df = trades_df.dropna(subset=['pct_gain_loss'])

        if trades_df.empty:
            raise ValueError("No completed trades to analyze.")

        # Portfolio Return (from portfolio value)
        portfolio_return = self.portfolio['portfolio_return_pct'].iloc[-1]
        
        # Trade Returns (sum of individual trade returns, exit trades only)
        exit_trades = trades_df[trades_df['signal'] == -1]
        trade_returns = exit_trades['pct_gain_loss'].sum()

        # Sharpe Ratio
        trades_df["periodic_return"] = trades_df["total_value"].pct_change().dropna()
        mean_return = trades_df['periodic_return'].mean()
        std_return = trades_df['periodic_return'].std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else np.nan

        # Max Drawdown (from cumulative portfolio returns)
        cumulative_returns = (trades_df['pct_gain_loss'] / 100 + 1).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak * 100
        max_drawdown = drawdown.min()
        max_drawdown_time = drawdown.idxmin()

        # Max trade return
        max_trade_return = trades_df['pct_gain_loss'].max()
        max_trade_return_time = trades_df['pct_gain_loss'].idxmax()

        # Number of trades
        num_trades = len(exit_trades) 

        # Win rate
        winning_trades = len(exit_trades[exit_trades['pct_gain_loss'] > 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        # Prepare metrics dictionary
        metrics = {
            'Portfolio Return': f"{portfolio_return:.2f}%",
            'Total Trade Returns': f"{trade_returns:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            #'Max Drawdown Occurred at': max_drawdown_time,
            'Max Trade Return': f"{max_trade_return:.2f}%",
            #'Max Trade Return Occurred at': max_trade_return_time,
            'Win Rate': f"{win_rate:.2f}%",
            'Number of Trades': num_trades
        }

        return metrics
    
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
    