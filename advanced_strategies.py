import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from stock_market import TradingStrategy, TradingSignal, TechnicalIndicators

class MeanReversionStrategy(TradingStrategy):
    def __init__(self, lookback_period: int = 20, zscore_threshold: float = 2.0):
        super().__init__("Mean Reversion Strategy")
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        if len(data) < self.lookback_period + 10:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")
        
        current_price = data['Close'].iloc[-1]
        rolling_mean = data['Close'].rolling(window=self.lookback_period).mean().iloc[-1]
        rolling_std = data['Close'].rolling(window=self.lookback_period).std().iloc[-1]
        
        if rolling_std == 0:
            return TradingSignal("HOLD", 0.0, current_price, datetime.now(), "No volatility")
        
        zscore = (current_price - rolling_mean) / rolling_std
        rsi = self.indicators.rsi(data['Close']).iloc[-1]
        
        confidence = min(abs(zscore) / self.zscore_threshold, 1.0) * 0.8
        
        if zscore < -self.zscore_threshold and rsi < 35:
            return TradingSignal("BUY", confidence, current_price, datetime.now(), 
                               f"Mean reversion buy signal (Z-score: {zscore:.2f})")
        elif zscore > self.zscore_threshold and rsi > 65:
            return TradingSignal("SELL", confidence, current_price, datetime.now(), 
                               f"Mean reversion sell signal (Z-score: {zscore:.2f})")
        
        return TradingSignal("HOLD", 0.0, current_price, datetime.now(), f"Z-score within range: {zscore:.2f}")

class MomentumStrategy(TradingStrategy):
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__("Momentum Strategy")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        if len(data) < self.long_window + 10:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")
        
        current_price = data['Close'].iloc[-1]
        short_ma = self.indicators.sma(data['Close'], self.short_window)
        long_ma = self.indicators.sma(data['Close'], self.long_window)
        rsi = self.indicators.rsi(data['Close'])
        
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        price_momentum = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
        
        if (prev_short <= prev_long and current_short > current_long and 
            current_rsi < 70 and price_momentum > 2):
            confidence = 0.8 if price_momentum > 5 else 0.6
            return TradingSignal("BUY", confidence, current_price, datetime.now(), 
                               f"Momentum crossover buy (momentum: {price_momentum:.1f}%)")
        elif (prev_short >= prev_long and current_short < current_long and 
              current_rsi > 30 and price_momentum < -2):
            confidence = 0.8 if price_momentum < -5 else 0.6
            return TradingSignal("SELL", confidence, current_price, datetime.now(), 
                               f"Momentum crossover sell (momentum: {price_momentum:.1f}%)")
        
        return TradingSignal("HOLD", 0.0, current_price, datetime.now(), "No momentum signal")

class BreakoutStrategy(TradingStrategy):
    def __init__(self, lookback_period: int = 20, volume_threshold: float = 1.5):
        super().__init__("Breakout Strategy")
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        if len(data) < self.lookback_period + 5:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")
        
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(window=self.lookback_period).mean().iloc[-1]
        
        high_20 = data['High'].rolling(window=self.lookback_period).max().iloc[-2]
        low_20 = data['Low'].rolling(window=self.lookback_period).min().iloc[-2]
        
        volume_surge = current_volume > (avg_volume * self.volume_threshold)
        
        upper_band, _, lower_band = self.indicators.bollinger_bands(data['Close'])
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if current_price > high_20 and volume_surge:
            confidence = 0.8 if current_price > current_upper else 0.6
            return TradingSignal("BUY", confidence, current_price, datetime.now(), 
                               f"Upward breakout with volume surge")
        elif current_price < low_20 and volume_surge:
            confidence = 0.8 if current_price < current_lower else 0.6
            return TradingSignal("SELL", confidence, current_price, datetime.now(), 
                               f"Downward breakout with volume surge")
        
        return TradingSignal("HOLD", 0.0, current_price, datetime.now(), "No breakout detected")

class MultiSignalStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("Multi-Signal Strategy")
        self.macd_strategy = None
        self.momentum_strategy = MomentumStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        if len(data) < 50:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")
        
        momentum_signal = self.momentum_strategy.generate_signal(data)
        mean_rev_signal = self.mean_reversion_strategy.generate_signal(data)
        
        macd_line, signal_line, _ = self.indicators.macd(data['Close'])
        rsi = self.indicators.rsi(data['Close'])
        
        current_price = data['Close'].iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0
        reasons = []
        
        if momentum_signal.signal == "BUY":
            buy_signals += 1
            total_confidence += momentum_signal.confidence
            reasons.append("Momentum bullish")
        elif momentum_signal.signal == "SELL":
            sell_signals += 1
            total_confidence += momentum_signal.confidence
            reasons.append("Momentum bearish")
        
        if mean_rev_signal.signal == "BUY":
            buy_signals += 1
            total_confidence += mean_rev_signal.confidence
            reasons.append("Mean reversion bullish")
        elif mean_rev_signal.signal == "SELL":
            sell_signals += 1
            total_confidence += mean_rev_signal.confidence
            reasons.append("Mean reversion bearish")
        
        if current_macd > current_signal and current_rsi < 70:
            buy_signals += 1
            total_confidence += 0.6
            reasons.append("MACD bullish")
        elif current_macd < current_signal and current_rsi > 30:
            sell_signals += 1
            total_confidence += 0.6
            reasons.append("MACD bearish")
        
        if buy_signals >= 2 and buy_signals > sell_signals:
            avg_confidence = min(total_confidence / max(buy_signals, 1), 1.0)
            return TradingSignal("BUY", avg_confidence, current_price, datetime.now(), 
                               f"Multi-signal buy: {', '.join(reasons)}")
        elif sell_signals >= 2 and sell_signals > buy_signals:
            avg_confidence = min(total_confidence / max(sell_signals, 1), 1.0)
            return TradingSignal("SELL", avg_confidence, current_price, datetime.now(), 
                               f"Multi-signal sell: {', '.join(reasons)}")
        
        return TradingSignal("HOLD", 0.0, current_price, datetime.now(), "Mixed or weak signals")

class VolatilityStrategy(TradingStrategy):
    def __init__(self, volatility_window: int = 20, volatility_threshold: float = 0.02):
        super().__init__("Volatility Strategy")
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        if len(data) < self.volatility_window + 10:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")
        
        current_price = data['Close'].iloc[-1]
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
        avg_volatility = returns.rolling(window=self.volatility_window * 2).std().mean()
        
        rsi = self.indicators.rsi(data['Close']).iloc[-1]
        atr = self.indicators.atr(data['High'], data['Low'], data['Close']).iloc[-1]
        
        volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        
        if volatility < self.volatility_threshold and rsi < 40:
            confidence = 0.7 if volatility_ratio < 0.8 else 0.5
            return TradingSignal("BUY", confidence, current_price, datetime.now(), 
                               f"Low volatility buy opportunity (vol: {volatility:.3f})")
        elif volatility > self.volatility_threshold * 2 and rsi > 60:
            confidence = 0.7 if volatility_ratio > 1.5 else 0.5
            return TradingSignal("SELL", confidence, current_price, datetime.now(), 
                               f"High volatility sell signal (vol: {volatility:.3f})")
        
        return TradingSignal("HOLD", 0.0, current_price, datetime.now(), 
                           f"Volatility neutral (vol: {volatility:.3f})")