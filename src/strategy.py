import pandas as pd
import pandas_ta as ta

class EMAStrategy:
    """
    Implements the EMA crossover trading strategy with multiple filters and confirmations.
    """
    
    def __init__(self, 
                 # EMA parameters
                 ema_short: int = 9, 
                 ema_long: int = 20,
                 use_ema: bool = True,
                 # ADX parameters
                 adx_period: int = 15,
                 adx_threshold: float = 25.0,
                 use_adx: bool = False,
                 # ATR parameters
                 atr_period: int = 14,
                 atr_threshold: float = 1.0,
                 use_atr: bool = False,
                 # RSI parameters
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 use_rsi: bool = False,
                 # MACD parameters
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 use_macd: bool = False):
        """
        Initialize the strategy with parameters for multiple indicators.
        
        Args:
            ema_short (int): Short-term EMA period
            ema_long (int): Long-term EMA period
            use_ema (bool): Whether to use EMA signals
            adx_period (int): ADX calculation period
            adx_threshold (float): ADX threshold for trend strength
            use_adx (bool): Whether to use ADX signals
            atr_period (int): ATR calculation period
            atr_threshold (float): ATR threshold for volatility
            use_atr (bool): Whether to use ATR signals
            rsi_period (int): RSI calculation period
            rsi_overbought (float): RSI overbought threshold
            rsi_oversold (float): RSI oversold threshold
            use_rsi (bool): Whether to use RSI signals
            macd_fast (int): MACD fast period
            macd_slow (int): MACD slow period
            macd_signal (int): MACD signal line period
            use_macd (bool): Whether to use MACD signals
        """
        # EMA parameters
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.use_ema = use_ema
        
        # ADX parameters
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_adx = use_adx
        
        # ATR parameters
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.use_atr = use_atr
        
        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_rsi = use_rsi
        
        # MACD parameters
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.use_macd = use_macd
        
        # Validate that at least one indicator is enabled
        if not any([use_ema, use_adx, use_atr, use_rsi, use_macd]):
            raise ValueError("At least one indicator must be enabled")

    def get_entry_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify entry conditions based on all enabled indicators.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with final trading signals
        """
        # Apply each enabled indicator
        if self.use_ema:
            data = self.generate_crossover_signals(data)
            
        if self.use_adx:
            data = self.ADX_filter(data)
            
        if self.use_atr:
            data = self.ATR_filter(data)
            
        if self.use_rsi:
            data = self.RSI_filter(data)
            
        if self.use_macd:
            data = self.MACD_filter(data)
        
        # Initialize final signal column
        data['signal'] = 0
        in_position = False
        
        # Combine signals according to enabled indicators
        for i in range(len(data)):
            # Collect all active signals
            signals = []
            
            if self.use_ema:
                signals.append(data.iloc[i]['signal_crossover'])
            if self.use_adx:
                signals.append(data.iloc[i]['signal_adx'])
            if self.use_atr:
                signals.append(data.iloc[i]['signal_atr'])
            if self.use_rsi:
                signals.append(data.iloc[i]['signal_rsi'])
            if self.use_macd:
                signals.append(data.iloc[i]['signal_macd'])
            
            if not in_position:
                # Enter position only if all enabled indicators show entry (1)
                if all(signal == 1 for signal in signals):
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    in_position = True
            else:
                # Exit position if any enabled indicator shows exit (-1)
                if any(signal == -1 for signal in signals):
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
    
    def ADX_filter(self, data: pd.DataFrame) -> pd.DataFrame:
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
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)[f'ADX_{self.adx_period}']
        df['ADX'] = df['ADX'].fillna(0)
        df['signal_adx'] = 0
        
        # Track position state while filtering
        in_position = False
        prev_adx = 0
        
        for i in range(len(df)):
            current_adx = df.iloc[i]['ADX']
            
            if not in_position:
                # Generate entry signal when ADX crosses above threshold
                if current_adx >= self.adx_threshold and prev_adx < self.adx_threshold:
                    df.iloc[i, df.columns.get_loc('signal_adx')] = 1
                    in_position = True
            else:
                # Generate exit signal when ADX drops below threshold
                if current_adx < self.adx_threshold and prev_adx >= self.adx_threshold:
                    df.iloc[i, df.columns.get_loc('signal_adx')] = -1
                    in_position = False
            
            prev_adx = current_adx
            
        return df
    
    def ATR_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ATR filter to generate volatility-based signals.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Data with ATR signals
        """
        df = data.copy()
        
        # Calculate ATR
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['ATR'] = df['ATR'].fillna(0)
        df['signal_atr'] = 0
        
        # Calculate ATR percentage of price
        df['ATR_pct'] = (df['ATR'] / df['close']) * 100
        
        in_position = False
        prev_atr_pct = 0
        
        for i in range(len(df)):
            current_atr_pct = df.iloc[i]['ATR_pct']
            
            if not in_position:
                # Enter when volatility increases above threshold
                if current_atr_pct >= self.atr_threshold and prev_atr_pct < self.atr_threshold:
                    df.iloc[i, df.columns.get_loc('signal_atr')] = 1
                    in_position = True
            else:
                # Exit when volatility drops below threshold
                if current_atr_pct < self.atr_threshold and prev_atr_pct >= self.atr_threshold:
                    df.iloc[i, df.columns.get_loc('signal_atr')] = -1
                    in_position = False
                    
            prev_atr_pct = current_atr_pct
            
        return df
    
    def RSI_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply RSI filter to generate overbought/oversold signals.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Data with RSI signals
        """
        df = data.copy()
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['close'], length=self.rsi_period)
        df['RSI'] = df['RSI'].fillna(0)
        df['signal_rsi'] = 0
        
        in_position = False
        
        for i in range(len(df)):
            current_rsi = df.iloc[i]['RSI']
            
            if not in_position:
                # Enter when RSI crosses above oversold
                if current_rsi > self.rsi_oversold:
                    df.iloc[i, df.columns.get_loc('signal_rsi')] = 1
                    in_position = True
            else:
                # Exit when RSI crosses above overbought
                if current_rsi > self.rsi_overbought:
                    df.iloc[i, df.columns.get_loc('signal_rsi')] = -1
                    in_position = False
                    
        return df
    
    def MACD_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MACD filter to generate trend signals.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Data with MACD signals
        """
        df = data.copy()
        
        # Calculate MACD
        macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df['MACD'] = macd['MACD_' + str(self.macd_fast) + '_' + str(self.macd_slow) + '_' + str(self.macd_signal)]
        df['MACD_Signal'] = macd['MACDs_' + str(self.macd_fast) + '_' + str(self.macd_slow) + '_' + str(self.macd_signal)]
        df['MACD_Hist'] = macd['MACDh_' + str(self.macd_fast) + '_' + str(self.macd_slow) + '_' + str(self.macd_signal)]
        
        df['signal_macd'] = 0
        in_position = False
        
        for i in range(1, len(df)):
            # Current and previous histogram values
            curr_hist = df.iloc[i]['MACD_Hist']
            prev_hist = df.iloc[i-1]['MACD_Hist']
            
            if not in_position:
                # Enter on histogram crossing above zero
                if curr_hist > 0 and prev_hist <= 0:
                    df.iloc[i, df.columns.get_loc('signal_macd')] = 1
                    in_position = True
            else:
                # Exit on histogram crossing below zero
                if curr_hist < 0 and prev_hist >= 0:
                    df.iloc[i, df.columns.get_loc('signal_macd')] = -1
                    in_position = False
                    
        return df