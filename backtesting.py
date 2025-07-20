import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf
from stock_market import SmartStockTrader, TradingSignal, Position
from advanced_strategies import (MomentumStrategy, MeanReversionStrategy, 
                               BreakoutStrategy, MultiSignalStrategy, VolatilityStrategy)

class Backtester:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def backtest_strategy(self, strategy, symbol: str, start_date: str, end_date: str) -> Dict:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                return {"error": f"No data found for {symbol}"}
            
            trader = SmartStockTrader(self.initial_capital)
            data = trader.add_indicators(data)
            
            portfolio_values = []
            trades = []
            position = None
            cash = self.initial_capital
            
            for i in range(50, len(data)):
                current_data = data.iloc[:i+1]
                current_price = current_data['Close'].iloc[-1]
                current_date = current_data.index[-1]
                
                signal = strategy.generate_signal(current_data)
                
                if signal.signal == "BUY" and position is None and signal.confidence > 0.6:
                    shares = int((cash * 0.95) / current_price)
                    if shares > 0:
                        cost = shares * current_price * (1 + self.commission)
                        if cost <= cash:
                            cash -= cost
                            position = {
                                'shares': shares,
                                'entry_price': current_price,
                                'entry_date': current_date
                            }
                            trades.append({
                                'action': 'BUY',
                                'shares': shares,
                                'price': current_price,
                                'date': current_date,
                                'confidence': signal.confidence
                            })
                
                elif signal.signal == "SELL" and position is not None and signal.confidence > 0.6:
                    proceeds = position['shares'] * current_price * (1 - self.commission)
                    cash += proceeds
                    
                    pnl = proceeds - (position['shares'] * position['entry_price'] * (1 + self.commission))
                    
                    trades.append({
                        'action': 'SELL',
                        'shares': position['shares'],
                        'price': current_price,
                        'date': current_date,
                        'pnl': pnl,
                        'confidence': signal.confidence,
                        'hold_days': (current_date - position['entry_date']).days
                    })
                    position = None
                
                portfolio_value = cash
                if position is not None:
                    portfolio_value += position['shares'] * current_price
                
                portfolio_values.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'price': current_price
                })
            
            if position is not None:
                final_price = data['Close'].iloc[-1]
                proceeds = position['shares'] * final_price * (1 - self.commission)
                cash += proceeds
                
                pnl = proceeds - (position['shares'] * position['entry_price'] * (1 + self.commission))
                trades.append({
                    'action': 'SELL',
                    'shares': position['shares'],
                    'price': final_price,
                    'date': data.index[-1],
                    'pnl': pnl,
                    'confidence': 1.0,
                    'hold_days': (data.index[-1] - position['entry_date']).days
                })
            
            final_value = cash
            total_return = (final_value - self.initial_capital) / self.initial_capital * 100
            
            portfolio_df = pd.DataFrame(portfolio_values)
            buy_hold_return = (data['Close'].iloc[-1] - data['Close'].iloc[50]) / data['Close'].iloc[50] * 100
            
            completed_trades = [t for t in trades if 'pnl' in t]
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            if len(portfolio_df) > 1:
                returns = portfolio_df['value'].pct_change().dropna()
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(portfolio_df['value'])
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            return {
                'strategy': strategy.name,
                'symbol': symbol,
                'period': f"{start_date} to {end_date}",
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return_pct': total_return,
                'buy_hold_return_pct': buy_hold_return,
                'excess_return_pct': total_return - buy_hold_return,
                'total_trades': len(completed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate_pct': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'trades': trades,
                'portfolio_values': portfolio_df
            }
            
        except Exception as e:
            return {"error": f"Backtesting failed: {str(e)}"}
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        peak = values.cummax()
        drawdown = (values - peak) / peak * 100
        return drawdown.min()
    
    def compare_strategies(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        strategies = {
            'Momentum': MomentumStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Breakout': BreakoutStrategy(),
            'Multi-Signal': MultiSignalStrategy(),
            'Volatility': VolatilityStrategy()
        }
        
        results = []
        
        for symbol in symbols:
            for strategy_name, strategy in strategies.items():
                print(f"Backtesting {strategy_name} on {symbol}...")
                result = self.backtest_strategy(strategy, symbol, start_date, end_date)
                
                if 'error' not in result:
                    results.append({
                        'Symbol': symbol,
                        'Strategy': strategy_name,
                        'Total Return (%)': result['total_return_pct'],
                        'Buy & Hold (%)': result['buy_hold_return_pct'],
                        'Excess Return (%)': result['excess_return_pct'],
                        'Win Rate (%)': result['win_rate_pct'],
                        'Total Trades': result['total_trades'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Max Drawdown (%)': result['max_drawdown_pct']
                    })
        
        return pd.DataFrame(results)
    
    def plot_backtest_results(self, result: Dict, show_trades: bool = True):
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        portfolio_df = result['portfolio_values']
        trades = result['trades']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(portfolio_df['date'], portfolio_df['value'], label='Portfolio Value', linewidth=2)
        ax1.axhline(y=result['initial_capital'], color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        
        if show_trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_values = [portfolio_df[portfolio_df['date'] <= date]['value'].iloc[-1] for date in buy_dates]
                ax1.scatter(buy_dates, buy_values, color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            
            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_values = [portfolio_df[portfolio_df['date'] <= date]['value'].iloc[-1] for date in sell_dates]
                ax1.scatter(sell_dates, sell_values, color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{result["strategy"]} - {result["symbol"]} Backtest Results')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        initial_price = portfolio_df['price'].iloc[0]
        normalized_prices = (portfolio_df['price'] / initial_price) * result['initial_capital']
        ax2.plot(portfolio_df['date'], normalized_prices, label=f'{result["symbol"]} (Buy & Hold)', color='orange', alpha=0.7)
        
        ax2.set_title('Strategy vs Buy & Hold Comparison')
        ax2.set_ylabel('Normalized Value ($)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n=== Backtest Summary ===")
        print(f"Strategy: {result['strategy']}")
        print(f"Symbol: {result['symbol']}")
        print(f"Period: {result['period']}")
        print(f"Initial Capital: ${result['initial_capital']:,.2f}")
        print(f"Final Value: ${result['final_value']:,.2f}")
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Buy & Hold Return: {result['buy_hold_return_pct']:.2f}%")
        print(f"Excess Return: {result['excess_return_pct']:.2f}%")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Win Rate: {result['win_rate_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")

def run_comprehensive_backtest():
    backtester = Backtester(initial_capital=100000)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print("Running comprehensive strategy comparison...")
    comparison_df = backtester.compare_strategies(symbols, start_date, end_date)
    
    print("\n=== Strategy Comparison Results ===")
    print(comparison_df.to_string(index=False))
    
    best_performers = comparison_df.groupby('Strategy')['Total Return (%)'].mean().sort_values(ascending=False)
    print(f"\n=== Average Performance by Strategy ===")
    for strategy, return_pct in best_performers.items():
        print(f"{strategy}: {return_pct:.2f}%")
    
    print(f"\nRunning detailed backtest for best strategy on AAPL...")
    best_strategy_name = best_performers.index[0]
    
    strategies = {
        'Momentum': MomentumStrategy(),
        'Mean Reversion': MeanReversionStrategy(),
        'Breakout': BreakoutStrategy(),
        'Multi-Signal': MultiSignalStrategy(),
        'Volatility': VolatilityStrategy()
    }
    
    best_strategy = strategies[best_strategy_name]
    detailed_result = backtester.backtest_strategy(best_strategy, 'AAPL', start_date, end_date)
    backtester.plot_backtest_results(detailed_result)
    
    return comparison_df, detailed_result

if __name__ == "__main__":
    run_comprehensive_backtest()