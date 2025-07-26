import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

@dataclass
class TradingSignal:
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    price: float
    timestamp: datetime
    reason: str

@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_date: datetime
    current_price: float = 0
    unrealized_pnl: float = 0

class TechnicalIndicators:
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = data.ewm(span=fast).mean()
        exp2 = data.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

class RiskManager:
    def __init__(self, max_position_size: float = 0.1, stop_loss_pct: float = 0.02, 
                 take_profit_pct: float = 0.06, max_daily_loss: float = 0.05):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
    
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float, 
                               entry_price: float, stop_loss_price: float) -> int:
        risk_amount = portfolio_value * risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            return 0
        shares = int(risk_amount / price_risk)
        max_shares = int((portfolio_value * self.max_position_size) / entry_price)
        return min(shares, max_shares)
    
    def should_exit_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        if pnl_pct <= -self.stop_loss_pct:
            return True, "STOP_LOSS"
        elif pnl_pct >= self.take_profit_pct:
            return True, "TAKE_PROFIT"
        return False, "HOLD"

class TradingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        raise NotImplementedError("Subclasses must implement generate_signal")

class MACDStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("MACD Strategy")
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        if len(data) < 50:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")
        
        macd_line, signal_line, histogram = self.indicators.macd(data['Close'])
        rsi = self.indicators.rsi(data['Close'])
        
        current_price = data['Close'].iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        confidence = 0.5
        signal = "HOLD"
        reason = "No clear signal"
        
        if (prev_macd <= prev_signal and current_macd > current_signal and 
            current_rsi < 70 and current_rsi > 30):
            signal = "BUY"
            confidence = 0.8 if current_rsi < 50 else 0.6
            reason = "MACD bullish crossover"
        elif (prev_macd >= prev_signal and current_macd < current_signal and 
              current_rsi > 30):
            signal = "SELL"
            confidence = 0.8 if current_rsi > 50 else 0.6
            reason = "MACD bearish crossover"
        
        return TradingSignal(signal, confidence, current_price, datetime.now(), reason)

class SmartStockTrader:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.risk_manager = RiskManager()
        self.strategies = {
            'macd': MACDStrategy()
        }
        self.logger = logging.getLogger(__name__)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        indicators = TechnicalIndicators()
        
        data['SMA_20'] = indicators.sma(data['Close'], 20)
        data['SMA_50'] = indicators.sma(data['Close'], 50)
        data['EMA_12'] = indicators.ema(data['Close'], 12)
        data['EMA_26'] = indicators.ema(data['Close'], 26)
        data['RSI'] = indicators.rsi(data['Close'])
        
        macd_line, signal_line, histogram = indicators.macd(data['Close'])
        data['MACD'] = macd_line
        data['MACD_Signal'] = signal_line
        data['MACD_Histogram'] = histogram
        
        upper_band, middle_band, lower_band = indicators.bollinger_bands(data['Close'])
        data['BB_Upper'] = upper_band
        data['BB_Middle'] = middle_band
        data['BB_Lower'] = lower_band
        
        data['ATR'] = indicators.atr(data['High'], data['Low'], data['Close'])
        
        return data
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            return self.add_indicators(data)
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_stock(self, symbol: str, strategy_name: str = 'macd') -> TradingSignal:
        data = self.get_stock_data(symbol)
        if data.empty:
            return TradingSignal("HOLD", 0.0, 0.0, datetime.now(), "No data available")
        
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        return strategy.generate_signal(data)
    
    def execute_trade(self, symbol: str, signal: TradingSignal, risk_per_trade: float = 0.02):
        if signal.signal == "HOLD":
            return
        
        current_data = self.get_stock_data(symbol, "5d")
        if current_data.empty:
            return
        
        current_price = signal.price
        atr = current_data['ATR'].iloc[-1]
        
        if signal.signal == "BUY" and symbol not in self.positions:
            stop_loss_price = current_price - (2 * atr)
            position_size = self.risk_manager.calculate_position_size(
                self.current_capital, risk_per_trade, current_price, stop_loss_price
            )
            
            if position_size > 0 and (position_size * current_price) <= self.current_capital:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=position_size,
                    entry_price=current_price,
                    entry_date=datetime.now(),
                    current_price=current_price
                )
                
                cost = position_size * current_price
                self.current_capital -= cost
                
                self.trade_history.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': position_size,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'confidence': signal.confidence,
                    'reason': signal.reason
                })
                
                self.logger.info(f"Bought {position_size} shares of {symbol} at ${current_price:.2f}")
        
        elif signal.signal == "SELL" and symbol in self.positions:
            position = self.positions[symbol]
            sale_proceeds = position.shares * current_price
            self.current_capital += sale_proceeds
            
            pnl = (current_price - position.entry_price) * position.shares
            
            self.trade_history.append({
                'symbol': symbol,
                'action': 'SELL',
                'shares': position.shares,
                'price': current_price,
                'timestamp': datetime.now(),
                'pnl': pnl,
                'confidence': signal.confidence,
                'reason': signal.reason
            })
            
            self.logger.info(f"Sold {position.shares} shares of {symbol} at ${current_price:.2f}, PnL: ${pnl:.2f}")
            del self.positions[symbol]
    
    def update_positions(self, symbols: List[str]):
        for symbol in symbols:
            if symbol in self.positions:
                current_data = self.get_stock_data(symbol, "1d")
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    position = self.positions[symbol]
                    position.current_price = current_price
                    position.unrealized_pnl = (current_price - position.entry_price) * position.shares
                    
                    should_exit, reason = self.risk_manager.should_exit_position(position, current_price)
                    if should_exit:
                        signal = TradingSignal("SELL", 1.0, current_price, datetime.now(), reason)
                        self.execute_trade(symbol, signal)
    
    def get_portfolio_value(self) -> float:
        total_value = self.current_capital
        for position in self.positions.values():
            total_value += position.shares * position.current_price
        return total_value
    
    def get_performance_metrics(self) -> Dict:
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len([t for t in self.trade_history if 'pnl' in t]) * 100 if self.trade_history else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'portfolio_value': portfolio_value,
            'total_return_pct': total_return,
            'total_trades': len([t for t in self.trade_history if 'pnl' in t]),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'positions': len(self.positions)
        }
    
    def plot_analysis(self, symbol: str, period: str = "6mo"):
        data = self.get_stock_data(symbol, period)
        if data.empty:
            print(f"No data available for {symbol}")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], alpha=0.2, label='Bollinger Bands')
        ax1.set_title(f'{symbol} Price Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(data.index, data['MACD'], label='MACD', color='blue')
        ax3.plot(data.index, data['MACD_Signal'], label='Signal', color='red')
        ax3.bar(data.index, data['MACD_Histogram'], label='Histogram', alpha=0.6, color='gray')
        ax3.set_title('MACD Indicator')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    trader = SmartStockTrader(initial_capital=100000)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("=== Smart Stock Market Trader ===")
    print(f"Initial Capital: ${trader.initial_capital:,.2f}")
    print("\nAnalyzing stocks...")
    
    for symbol in symbols:
        print(f"\n--- {symbol} Analysis ---")
        signal = trader.analyze_stock(symbol)
        print(f"Signal: {signal.signal}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Price: ${signal.price:.2f}")
        print(f"Reason: {signal.reason}")
        
        if signal.confidence > 0.6:
            trader.execute_trade(symbol, signal)
    
    trader.update_positions(symbols)
    
    print("\n=== Portfolio Performance ===")
    metrics = trader.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    if symbols:
        print(f"\nGenerating analysis chart for {symbols[0]}...")
        trader.plot_analysis(symbols[0])

if __name__ == "__main__":
    main()
