# AI Agent Instructions for EMA Backtest Project

## Project Overview
This is a quantitative trading backtesting system that implements an Exponential Moving Average (EMA) crossover strategy. The system uses modular components to handle data management, strategy implementation, and backtesting execution.

## Core Architecture

### Key Components
1. **DataHandler** (`src/data_handler.py`)
   - Manages data fetching from Alpaca API with CSV caching
   - Handles data validation and technical indicator calculations
   - Implements rate limiting (200 requests/minute)

2. **EMAStrategy** (`src/strategy.py`)
   - Implements EMA crossover strategy (9/20 period default)

3. **Backtester** (`src/backtester.py`)
   - Executes trading strategy simulation
   - Maintains portfolio state and trade history
   - Calculates performance metrics

## Key Workflows

### Data Pipeline
```python
data_handler = DataHandler(API_KEY, SECRET_KEY, symbol="AMD")
data = data_handler.fetch_data(
    timeframe=TimeFrame.Hour,
    start=datetime(2022, 1, 1),
    end=datetime(2025, 8, 1)
)
data = data_handler.add_technical_indicators(ema_short=9, ema_long=20)
```

### Backtesting Flow
```python
strategy = EMAStrategy(ema_short=9, ema_long=20)
backtester = Backtester(strategy, initial_capital=10000.0)
portfolio = backtester.run(data)
```

## Critical Patterns

### Strategy Implementation
- Use `generate_signals` to create buy/sell signals based on EMA crossovers

### Data Handling
- Always validate data for missing values and duplicates
- Use CSV caching to minimize API calls
- Handle rate limits with built-in checks

### Error Handling
- Validate trade execution conditions
- Track and log portfolio state changes
- Monitor unexplained P&L discrepancies

## Integration Points
1. Alpaca API integration for market data
2. CSV file system for data caching
3. Jupyter notebooks for analysis visualization

## Development Guidelines
1. Always run full backtest after strategy changes
2. Monitor trade logs for unexpected portfolio value changes
3. Use the notebook's analysis tools for strategy validation
4. Keep API rate limits in mind during development

## Common Pitfalls
1. Not accounting for slippage in backtests
2. Ignoring API rate limits
3. Missing trade exit conditions
4. Not validating portfolio value changes

For questions about implementation details, refer to the example notebook `ema_backtest.ipynb`.
