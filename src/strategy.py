import pandas as pd
import pandas_ta as ta

class EMAStrategy:
    """
    Implements the EMA crossover trading strategy with multiple filters and confirmations.
    """
    
    def __init__(self, 
                 # EMA parameters
                 ema_short: int, 
                 ema_long: int,
                 use_ema: bool = True,
                 # ADX parameters
                 adx_period: int = 15,
                 adx_threshold: float = 15.0,
                 use_adx: bool = False,
                 # ATR parameters
                 atr_period: int = 14,
                 atr_threshold: float = 1.0,
                 use_atr: bool = True,
                 # RSI parameters
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 use_rsi: bool = False):
        """
        Initialize the strategy with parameters for multiple indicators.
        
        Args:
            ema_short (int): Short-term EMA period
            ema_long (int): Long-term EMA period
            use_ema (bool): Whether to use EMA signals (mandatory for strategy)
            adx_period (int): ADX calculation period
            adx_threshold (float): ADX threshold for trend strength
            use_adx (bool): Whether to use ADX as entry filter
            atr_period (int): ATR calculation period
            atr_threshold (float): ATR threshold for volatility
            use_atr (bool): Whether to use ATR for exits
            rsi_period (int): RSI calculation period
            rsi_overbought (float): RSI overbought threshold for blocking entries
            use_rsi (bool): Whether to use RSI as entry filter
        """
        # EMA parameters (mandatory)
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.use_ema = use_ema
        
        # ADX parameters (optional entry filter)
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_adx = use_adx
        
        # ATR parameters (optional exit filter)
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.use_atr = use_atr
        
        # RSI parameters (optional entry filter)
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.use_rsi = use_rsi
        
        # Validate that EMA is enabled (mandatory)
        if not use_ema:
            raise ValueError("EMA must be enabled as it is the core strategy")

    def get_entry_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify entry and exit conditions based on EMA crossover and filters.
        EMA provides core trade signals. ADX and RSI act as entry filters only.
        ATR is used only for exits.
        
        Args:
            data (pd.DataFrame): Market data with technical indicators
            
        Returns:
            pd.DataFrame: Data with final trading signals
        """
        # Generate EMA signals (mandatory)
        data = self.generate_crossover_signals(data)
        
        # Apply optional filters
        if self.use_adx:
            data = self.ADX_filter(data)
        if self.use_atr:
            data = self.ATR_filter(data)
        if self.use_rsi:
            data = self.RSI_filter(data)
        
        # Initialize final signal column
        data['signal'] = 0
        in_position = False
        
        for i in range(len(data)):
            # Get EMA signal
            ema_signal = data.iloc[i]['signal_crossover']
            
            # Check entry blocks from filters
            entry_blocked = False
            
            if self.use_adx:
                current_adx = data.iloc[i]['ADX']
                if current_adx < self.adx_threshold:
                    entry_blocked = True
                    
            if self.use_rsi:
                current_rsi = data.iloc[i]['RSI']
                if current_rsi > self.rsi_overbought:
                    entry_blocked = True
            
            if not in_position:
                # Enter only if EMA signals entry and no filter blocks it
                if ema_signal == 1 and not entry_blocked:
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    in_position = True
            else:
                # Check exit conditions
                should_exit = False
                
                # EMA exit signal
                if ema_signal == -1:
                    should_exit = True
                    
                # ATR exit condition if enabled
                if self.use_atr:
                    atr_signal = data.iloc[i]['signal_atr']
                    if atr_signal == -1:
                        should_exit = True
                
                if should_exit:
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
            short_gt_long = df['EMA_short'].iloc[i] > df['EMA_long'].iloc[i]
            prev_short_gt_long = df['EMA_short'].iloc[i-1] > df['EMA_long'].iloc[i-1]

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
        Calculate ADX indicator for entry filtering.
        Entries are blocked when ADX is below threshold.
        
        Args:
            data (pd.DataFrame): Market data with ADX indicator
            
        Returns:
            pd.DataFrame: Data with ADX values
        """
        df = data.copy()
        
        # Calculate ADX
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)[f'ADX_{self.adx_period}']
        df['ADX'] = df['ADX'].fillna(0)
        
        return df
    
    def ATR_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ATR filter for exit signals only.
        Generates exit signals based on ATR volatility threshold.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Data with ATR signals
        """
        df = data.copy()
        
        # Calculate ATR
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['ATR'] = df['ATR'].fillna(0)
        
        # Calculate ATR percentage of price
        df['ATR_pct'] = (df['ATR'] / df['close']) * 100
        df['signal_atr'] = 0
        
        # Only track ATR for exits
        prev_atr_pct = 0
        for i in range(len(df)):
            current_atr_pct = df.iloc[i]['ATR_pct']
            
            # Generate exit signal when volatility crosses threshold
            if current_atr_pct >= self.atr_threshold and prev_atr_pct < self.atr_threshold:
                df.iloc[i, df.columns.get_loc('signal_atr')] = -1
                    
            prev_atr_pct = current_atr_pct
            
        return df
    
    def RSI_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI indicator for entry filtering.
        Entries are blocked when RSI is above overbought level.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Data with RSI values
        """
        df = data.copy()
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['close'], length=self.rsi_period)
        df['RSI'] = df['RSI'].fillna(0)
        
        return df
    
