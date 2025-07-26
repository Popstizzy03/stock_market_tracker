# Smart Stock Market Trader

An intelligent, powerful, and reliable stock market trading system with advanced technical analysis, risk management, and automated trading capabilities.

## Features

### Core Trading System
- **Smart Stock Trader**: Comprehensive trading system with portfolio management
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA, ATR
- **Risk Management**: Position sizing, stop-loss, take-profit, daily loss limits
- **Multiple Trading Strategies**: MACD, Momentum, Mean Reversion, Breakout, Volatility, Multi-Signal

### Advanced Capabilities
- **Backtesting Engine**: Test strategies on historical data with detailed performance metrics
- **Real-Time Trading**: Automated trading with market hours detection and alert system
- **Portfolio Analytics**: Performance tracking, win rates, Sharpe ratio, max drawdown
- **Strategy Comparison**: Compare multiple strategies across different stocks

## File Structure

```
stock_market_tracker/
├── stock_market.py          # Core trading system and technical indicators
├── advanced_strategies.py   # Additional trading strategies
├── backtesting.py          # Backtesting engine and strategy comparison
├── real_time_trader.py     # Real-time trading with alerts and monitoring
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_market_tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from stock_market import SmartStockTrader

# Initialize trader with $100,000 capital
trader = SmartStockTrader(initial_capital=100000)

# Analyze a stock
signal = trader.analyze_stock('AAPL')
print(f"Signal: {signal.signal}, Confidence: {signal.confidence}")

# Execute trade based on signal
trader.execute_trade('AAPL', signal)

# Get performance metrics
metrics = trader.get_performance_metrics()
print(f"Total Return: {metrics['total_return_pct']:.2f}%")

# Plot technical analysis
trader.plot_analysis('AAPL')
```

### Backtesting
```python
from backtesting import run_comprehensive_backtest

# Run comprehensive backtest comparing all strategies
comparison_df, detailed_result = run_comprehensive_backtest()
```

### Real-Time Trading
```python
from real_time_trader import RealTimeTrader

# Create configuration and start trading
trader = RealTimeTrader()

# Run in simulation mode for 1 hour
trader.run_simulation_mode(1)

# Or start real trading (use with caution!)
# trader.start_trading()
```

## Trading Strategies

### 1. MACD Strategy
- Uses MACD crossovers combined with RSI filtering
- Generates buy signals on bullish crossovers with RSI < 70
- Generates sell signals on bearish crossovers with RSI > 30

### 2. Momentum Strategy
- Based on moving average crossovers and price momentum
- Identifies trending markets with strong directional movement
- Includes momentum filtering to avoid false signals

### 3. Mean Reversion Strategy
- Uses Z-score analysis to identify oversold/overbought conditions
- Combined with RSI for additional confirmation
- Effective in ranging markets

### 4. Breakout Strategy
- Detects price breakouts from recent highs/lows
- Requires volume confirmation for signal validation
- Uses Bollinger Bands for additional context

### 5. Volatility Strategy
- Trades based on volatility patterns
- Buys during low volatility with oversold conditions
- Sells during high volatility with overbought conditions

### 6. Multi-Signal Strategy
- Combines multiple strategies for robust signal generation
- Requires consensus from multiple indicators
- Higher confidence leads to better performance

## Risk Management

The system includes comprehensive risk management features:

- **Position Sizing**: Automatically calculates optimal position sizes based on risk tolerance
- **Stop Loss**: 2% default stop loss with ATR-based dynamic stops
- **Take Profit**: 6% default take profit levels
- **Daily Loss Limits**: Prevents excessive losses in a single day
- **Maximum Positions**: Limits the number of concurrent positions

## Performance Metrics

The system tracks detailed performance metrics:

- Total return and excess return vs buy-and-hold
- Win rate and profit factor
- Sharpe ratio for risk-adjusted returns
- Maximum drawdown analysis
- Trade-by-trade analysis

## Real-Time Features

### Market Hours Detection
- Automatically detects market hours (9:30 AM - 4:00 PM EST)
- Skips trading during weekends and after hours

### Alert System
- Daily loss limit alerts
- Position-specific profit/loss alerts
- Configurable alert thresholds

### Automated Reporting
- Daily performance reports
- Position summaries
- Trade logging with timestamps

## Configuration

Real-time trading can be customized via `trader_config.json`:

```json
{
  "initial_capital": 100000,
  "watchlist": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
  "trading_hours": {"start": "09:30", "end": "16:00"},
  "risk_per_trade": 0.02,
  "max_positions": 5,
  "strategies": ["multi_signal", "momentum"],
  "alert_thresholds": {
    "daily_loss_limit": 0.05,
    "position_loss_limit": 0.03,
    "profit_target": 0.08
  },
  "update_interval_minutes": 15
}
```

## Safety Features

- **Simulation Mode**: Test strategies without real money
- **Data Validation**: Comprehensive error handling for market data
- **Logging**: Detailed logging of all trading activities
- **Risk Limits**: Multiple layers of risk management
- **Manual Override**: Easy start/stop controls

## Example Results

The backtesting system can show results like:

```
Strategy Comparison Results:
Symbol    Strategy         Total Return (%)  Win Rate (%)  Sharpe Ratio
AAPL      Multi-Signal     15.2             67.5          1.23
AAPL      Momentum         12.8             58.3          0.98
AAPL      Mean Reversion   8.5              71.2          0.87
```

## Disclaimer

This software is for educational and research purposes only. Trading stocks involves risk and you can lose money. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
