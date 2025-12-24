
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf
from rich.console import Console
from rich.table import Table

from src.core.trader import SmartStockTrader, TradingSignal, Position
from src.strategies.advanced_strategies import (MomentumStrategy, MeanReversionStrategy,
                               BreakoutStrategy, MultiSignalStrategy, VolatilityStrategy)
from src.strategies.ml_strategy import MLTradingStrategy

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
        console = Console()
        strategies = {
            'Momentum': MomentumStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Breakout': BreakoutStrategy(),
            'Multi-Signal': MultiSignalStrategy(),
            'Volatility': VolatilityStrategy(),
            'ML Strategy': MLTradingStrategy()
        }
        
        results = []
        
        for symbol in symbols:
            for strategy_name, strategy in strategies.items():
                console.print(f"Backtesting [bold]{strategy_name}[/bold] on [bold]{symbol}[/bold]...")
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

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=(f'{result["strategy"]} - {result["symbol"]} Backtest Results',
                                            'Strategy vs Buy & Hold Comparison'))

        # Portfolio Value
        fig.add_trace(go.Scatter(x=portfolio_df['date'], y=portfolio_df['value'], name='Portfolio Value'), row=1, col=1)
        fig.add_hline(y=result['initial_capital'], line_dash="dash", line_color="grey", row=1, col=1)

        if show_trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_values = [portfolio_df[portfolio_df['date'] <= date]['value'].iloc[-1] for date in buy_dates]
                fig.add_trace(go.Scatter(x=buy_dates, y=buy_values, mode='markers',
                                         marker=dict(color='green', symbol='triangle-up', size=10),
                                         name='Buy Signal'), row=1, col=1)

            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_values = [portfolio_df[portfolio_df['date'] <= date]['value'].iloc[-1] for date in sell_dates]
                fig.add_trace(go.Scatter(x=sell_dates, y=sell_values, mode='markers',
                                         marker=dict(color='red', symbol='triangle-down', size=10),
                                         name='Sell Signal'), row=1, col=1)

        # Buy & Hold Comparison
        initial_price = portfolio_df['price'].iloc[0]
        normalized_prices = (portfolio_df['price'] / initial_price) * result['initial_capital']
        fig.add_trace(go.Scatter(x=portfolio_df['date'], y=normalized_prices, name=f'{result["symbol"]} (Buy & Hold)',
                                 line_color='orange'), row=2, col=1)

        fig.update_layout(height=800, title_text=f"Backtest Analysis for {result['symbol']}", showlegend=True)
        fig.show()
        
        console = Console()
        table = Table(title="Backtest Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")

        table.add_row("Strategy", result['strategy'])
        table.add_row("Symbol", result['symbol'])
        table.add_row("Period", result['period'])
        table.add_row("Initial Capital", f"${result['initial_capital']:,.2f}")
        table.add_row("Final Value", f"${result['final_value']:,.2f}")
        table.add_row("Total Return", f"{result['total_return_pct']:.2f}%")
        table.add_row("Buy & Hold Return", f"{result['buy_hold_return_pct']:.2f}%")
        table.add_row("Excess Return", f"{result['excess_return_pct']:.2f}%")
        table.add_row("Total Trades", str(result['total_trades']))
        table.add_row("Win Rate", f"{result['win_rate_pct']:.2f}%")
        table.add_row("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
        table.add_row("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")
        
        console.print(table)

def run_comprehensive_backtest():
    console = Console()
    backtester = Backtester(initial_capital=100000)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    console.print("[bold green]Running comprehensive strategy comparison...[/bold green]")
    comparison_df = backtester.compare_strategies(symbols, start_date, end_date)
    
    table = Table(title="Strategy Comparison Results")
    for col in comparison_df.columns:
        table.add_column(col, justify="right", style="cyan", no_wrap=True)

    for index, row in comparison_df.iterrows():
        table.add_row(*[f"{val:.2f}" if isinstance(val, float) else str(val) for val in row])

    console.print(table)
    
    best_performers = comparison_df.groupby('Strategy')['Total Return (%)'].mean().sort_values(ascending=False)

    avg_table = Table(title="Average Performance by Strategy")
    avg_table.add_column("Strategy", style="magenta")
    avg_table.add_column("Average Return (%)", justify="right", style="green")

    for strategy, return_pct in best_performers.items():
        avg_table.add_row(strategy, f"{return_pct:.2f}%")

    console.print(avg_table)

    console.print(f"\n[bold green]Running detailed backtest for best strategy on AAPL...[/bold green]")
    
    best_strategy_name = best_performers.index[0]
    
    strategies = {
        'Momentum': MomentumStrategy(),
        'Mean Reversion': MeanReversionStrategy(),
        'Breakout': BreakoutStrategy(),
        'Multi-Signal': MultiSignalStrategy(),
        'Volatility': VolatilityStrategy(),
        'ML Strategy': MLTradingStrategy()
    }
    
    best_strategy = strategies[best_strategy_name]
    detailed_result = backtester.backtest_strategy(best_strategy, 'AAPL', start_date, end_date)
    backtester.plot_backtest_results(detailed_result)
    
    return comparison_df, detailed_result

if __name__ == "__main__":
    run_comprehensive_backtest()
