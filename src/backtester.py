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
                 initial_capital: float,
                 position_size: float
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
        self.position_size = position_size
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

         # Initialize portfolio tracking
        self.portfolio = self._initialize_portfolio(data)

        in_position = False
        entry_price = 0.0
        shares = 0
        cash = self.initial_capital
        total_value = self.initial_capital
        position_size = self.position_size 
        trade_pnl = 0

        for i in range(len(self.portfolio)):
            current_price = self.portfolio.iloc[i]['close']
            signal = self.portfolio.iloc[i]['signal']

            if not in_position:
                # Enter long on signal 1 if we have cash
                if signal == 1 and cash >= position_size:
                    shares = position_size / current_price
                    entry_price = current_price
                    cash -= shares * current_price
                    in_position = True
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('trade_price')] = entry_price
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('shares')] = round(shares, 2)

            else:
                # Exit long on signal -1
                if signal == -1:
                    # Calculate trade P&L
                    trade_pnl = (current_price - entry_price) * shares
                    cash += shares * current_price

                    # Record exit details
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('trade_price')] = current_price
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('pnl_trade')] = round(trade_pnl, 2)
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('win_loss')] = 1 if trade_pnl > 0 else 0
                    self.portfolio.iloc[i, self.portfolio.columns.get_loc('shares')] = 0

                     # Reset position tracking
                    shares = 0
                    entry_price = 0.0
                    in_position = False
                    
                    

            # Update holdings and total portfolio value on every step
            holdings = shares * current_price
            total_value = cash + holdings

            # Update portfolio metrics
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('cash')] = round(cash, 2)
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('holdings')] = round(holdings, 2)
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('total_value')] = round(total_value, 2)
            self.portfolio.iloc[i, self.portfolio.columns.get_loc('portfolio_return_pct')] = round(
                (total_value - self.initial_capital) / self.initial_capital * 100, 2
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
        portfolio['trade_price'] = 0.0
        portfolio['shares'] = 0
        portfolio['pnl_trade'] = 0.0
        portfolio['win_loss'] = 0.0
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
                buy_trade['value'] = self.position_size  # fixed buy value
                buy_trade['pct_gain_loss'] = 0  # Entry trades have no gain/loss
                trades_list.append(buy_trade)
                open_trade = buy_trade
            elif row['signal'] == -1 and open_trade is not None:  # Sell
                # Record the sell
                sell_trade = row.copy()
                sell_trade['value'] = row['trade_price'] * row['shares']  # actual sell value
                # Calculate pct gain/loss relative to the last buy
                if open_trade['trade_price'] != 0:
                    pct_gain_loss = round(((row['trade_price'] - open_trade['trade_price']) / open_trade['trade_price']) * 100, 2)
                    sell_trade['pct_gain_loss'] = pct_gain_loss  # Only exit trades have gain/loss
                # Set holdings to 0 after sell
                sell_trade['holdings'] = 0
                trades_list.append(sell_trade)
                open_trade = None

        # Create final DataFrame
        trades_df = pd.DataFrame(trades_list)

        # Store internally
        self.trades = trades_df

        return self.trades
