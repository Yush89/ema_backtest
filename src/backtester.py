from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .strategy import EMAStrategy
from .data_handler import DataHandler

class Backtester:
    """
    Main backtesting engine for evaluating trading strategies.
    """
    
    def __init__(self,
                 strategy: EMAStrategy,
                 initial_capital: float = 10000.0
                 ):
        """
        Initialize the backtester.
        
        Args:
            strategy (EMAStrategy): Trading strategy instance
            initial_capital (float): Initial capital for backtesting
            commission (float): Commission rate per trade
            slippage (float): Slippage rate per trade
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = None
        self.trades = None


    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest.

        Args:
            data (pd.DataFrame): Must contain ['timestamp', 'close', 'EMA_9', 'EMA_20', 'signal']

        Returns:
            pd.DataFrame: Portfolio performance over time.
        """
        initial_capital = 10000
        position_size = 500  # fixed capital per trade

         # Initialize portfolio tracking
        self.portfolio = self._initialize_portfolio(data)

        in_position = False
        entry_price = 0.0
        shares = 0
        cash = initial_capital
        total_value = initial_capital
        last_trade_pnl = 0.0

        for i in range(len(self.portfolio)):
            price = self.portfolio.iloc[i]['close']
            signal = self.portfolio.iloc[i]['signal']

            if not in_position:
                # Enter long on signal 1 if we have cash
                if signal == 1 and cash >= position_size:
                    shares = position_size / price
                    entry_price = price
                    cash -= shares * price
                    in_position = True
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('trade_price')] = price
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('shares')] = round(shares, 2)

            else:
                # Exit long on signal -1
                if signal == -1:
                    cash += shares * price
                    last_trade_pnl = (price - entry_price) * shares
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('pnl_trade')] = last_trade_pnl
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('win_loss')] = 1 if last_trade_pnl > 0 else 0
                    shares = 0
                    in_position = False
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('trade_price')] = price
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('shares')] = 0

            # Update holdings and total portfolio value on every step
            holdings = shares * price
            total_value = cash + holdings
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('cash')] = round(cash, 2)
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('holdings')] = holdings
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('total_value')] = round(total_value, 2)
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('portfolio_return_pct')] = (
                (total_value - initial_capital) / initial_capital * 100
            )

        return self.portfolio
    
    def _initialize_portfolio(self, data : pd.DataFrame) -> pd.DataFrame:
        """
        Initialize the portfolio DataFrame.
        
        Args:
            index (pd.DatetimeIndex): Time index for the portfolio
            
        Returns:
            pd.DataFrame: Initialized portfolio DataFrame
        """
        portfolio = data.copy()
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total_value'] = self.initial_capital
        portfolio['trade_price'] = np.nan
        portfolio['shares'] = 0
        portfolio['pnl_trade'] = np.nan
        portfolio['win_loss'] = np.nan
        portfolio['portfolio_return_pct'] = 0.0

        portfolio = portfolio.astype({
            'shares': 'float',
            'cash': 'float',
            'total_value': 'float'
            })
        return portfolio

    def log_trades(self) -> pd.DataFrame:
        """
        Create a trades DataFrame from portfolio rows where trades occurred.
        Returns:
            pd.DataFrame: DataFrame of trades
        """
        if self.portfolio is None or self.portfolio.empty:
            raise ValueError("Portfolio is empty. Run the strategy before logging trades.")

        # Filter rows where a trade actually happened
        trade_rows = self.portfolio[self.portfolio['signal'] != 0].copy()

        trades_list = []
        open_trade = None

        for _, row in trade_rows.iterrows():
            if row['signal'] == 1:  # Buy
                # Record the buy
                buy_trade = row.copy()
                buy_trade['value'] = 500  # fixed buy value
                buy_trade['pct_gain_loss'] = None
                trades_list.append(buy_trade)
                open_trade = buy_trade
            elif row['signal'] == -1 and open_trade is not None:  # Sell
                # Record the sell
                sell_trade = row.copy()
                sell_trade['value'] = row['trade_price'] * row['shares']  # actual sell value
                # Calculate pct gain/loss relative to the last buy
                pct_gain_loss = ((row['trade_price'] - open_trade['trade_price']) / open_trade['trade_price']) * 100
                sell_trade['pct_gain_loss'] = pct_gain_loss
                # Update the last buy with the same pct_gain_loss
                trades_list[-1]['pct_gain_loss'] = pct_gain_loss
                # Set holdings to 0 after sell
                sell_trade['holdings'] = 0
                trades_list.append(sell_trade)
                open_trade = None

        # Create final DataFrame
        trades_df = pd.DataFrame(trades_list)

        # Store internally
        self.trades = trades_df

        return self.trades
