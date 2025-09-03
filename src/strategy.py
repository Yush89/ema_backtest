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
        Identify entry conditions based on EMA crossover and ADX filter confirmation.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with final trading signals
        """
        # Generate individual signals
        data = self.generate_crossover_signals(data)
        data = self.ADX_filter(data, adx_threshold=25)
        
        # Initialize final signal column
        data['signal'] = 0
        in_position = False
        
        # Combine signals according to rules
        for i in range(len(data)):
            crossover_signal = data.iloc[i]['signal_crossover']
            adx_signal = data.iloc[i]['signal_adx']
            
            if not in_position:
                # Enter position only if both signals are 1
                if crossover_signal == 1 and adx_signal == 1:
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    in_position = True
            else:
                # Exit position if either signal is -1
                if crossover_signal == -1 or adx_signal == -1:
                    data.iloc[i, data.columns.get_loc('signal')] = -1
                    in_position = False
        
        return data
        
    def generate_crossover_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based purely on EMA crossover.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with crossover signals
        """
        df = data.copy()
        df['signal_crossover'] = 0
        in_position = False

        # Generate base crossover signals
        for i in range(1, len(df)):
            # Check crossover conditions
            short_gt_long = df[f'EMA_{self.ema_short}'].iloc[i] > df[f'EMA_{self.ema_long}'].iloc[i]
            prev_short_gt_long = df[f'EMA_{self.ema_short}'].iloc[i-1] > df[f'EMA_{self.ema_long}'].iloc[i-1]

            if not in_position:
                # Entry signal on bullish crossover
                if short_gt_long and not prev_short_gt_long:
                    df.iloc[i, df.columns.get_loc('signal_crossover')] = 1
                    in_position = True
            else:
                # Exit signal on bearish crossover
                if not short_gt_long and prev_short_gt_long:
                    df.iloc[i, df.columns.get_loc('signal_crossover')] = -1
                    in_position = False
                else:
                    # Keep signal at 0 while in position
                    df.iloc[i, df.columns.get_loc('signal_crossover')] = 0
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
        
        # Calculate ADX
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=15)['ADX_15']
        df['ADX'] = df['ADX'].fillna(0)
        df['signal_adx'] = 0
        
        # Track position state while filtering
        in_position = False
        prev_adx = 0
        
        for i in range(len(df)):
            current_adx = df.iloc[i]['ADX']
            
            if not in_position:
                # Generate entry signal when ADX crosses above threshold
                if current_adx >= adx_threshold and prev_adx < adx_threshold:
                    df.iloc[i, df.columns.get_loc('signal_adx')] = 1
                    in_position = True
            else:
                # Generate exit signal when ADX drops below threshold
                if current_adx < adx_threshold and prev_adx >= adx_threshold:
                    df.iloc[i, df.columns.get_loc('signal_adx')] = -1
                    in_position = False
            
            prev_adx = current_adx
            
        return df