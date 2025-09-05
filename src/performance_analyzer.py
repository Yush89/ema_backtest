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
        Generate a detailed trade report showing complete trades (entry and exit pairs).
        
        Returns:
            pd.DataFrame: Trade report with performance metrics including:
                - Entry and exit prices
                - Profit/Loss per trade
                - Percentage return per trade
                - Trade duration
                - Total value at trade exit
        """
        trades_df = self.trades.copy()
        
        # Initialize lists to store complete trade information
        complete_trades = []
        entry_price = None
        entry_time = None
        entry_shares = None
        
        for idx, row in trades_df.iterrows():
            if row['signal'] == 1:  # Entry
                entry_price = row['trade_price']
                entry_time = idx
                entry_shares = row['shares']
            elif row['signal'] == -1 and entry_price is not None:  # Exit
                exit_price = row['trade_price']
                
                # Calculate trade metrics
                profit = (exit_price - entry_price) * entry_shares
                pct_return = (exit_price - entry_price) / entry_price * 100
                duration = idx - entry_time
                
                complete_trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': entry_shares,
                    'profit': profit,
                    'return_pct': pct_return,
                    'duration': duration,
                    'total_value': row['total_value']
                })
                
                # Reset entry variables
                entry_price = None
                entry_time = None
                entry_shares = None
        
        # Convert to DataFrame
        report_df = pd.DataFrame(complete_trades)
        
        if not report_df.empty:
            # Set index to exit time for consistency with other methods
            report_df.set_index('exit_time', inplace=True)
            
            # Add cumulative metrics
            report_df['cumulative_profit'] = report_df['profit'].cumsum()
            report_df['cumulative_return_pct'] = report_df['return_pct'].cumsum()
            
            # Convert duration to days if it's a timedelta
            if 'duration' in report_df.columns and isinstance(report_df['duration'].iloc[0], pd.Timedelta):
                report_df['duration'] = report_df['duration'].dt.total_seconds() / (24 * 60 * 60)  # Convert to days
            
            # Force numeric type and round all numeric columns to 2 decimal places
            numeric_columns = ['entry_price', 'exit_price', 'shares', 'profit', 'return_pct', 
                             'total_value', 'cumulative_profit', 'cumulative_return_pct', 'duration']
            
            for col in numeric_columns:
                if col in report_df.columns:
                    try:
                        report_df[col] = pd.to_numeric(report_df[col])
                    except (ValueError, TypeError):
                        # Skip columns that can't be converted to numeric
                        continue
                    report_df[col] = report_df[col].round(2)
            
            # Ensure the DataFrame uses the rounded values for display
            report_df = report_df.round(2)
        
        return report_df
    