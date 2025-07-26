import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from stock_market import SmartStockTrader, TradingSignal
from advanced_strategies import MultiSignalStrategy, MomentumStrategy, MeanReversionStrategy
import json
import os

class RealTimeTrader:
    def __init__(self, config_file: str = "trader_config.json"):
        self.config = self._load_config(config_file)
        self.trader = SmartStockTrader(self.config.get('initial_capital', 100000))
        self.active_strategies = self._initialize_strategies()
        self.watchlist = self.config.get('watchlist', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'])
        self.trading_active = False
        self.last_update = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.performance_log = []
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'daily_loss_limit': 0.05,
            'position_loss_limit': 0.03,
            'profit_target': 0.08
        })
    
    def _load_config(self, config_file: str) -> Dict:
        default_config = {
            'initial_capital': 100000,
            'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'trading_hours': {'start': '09:30', 'end': '16:00'},
            'risk_per_trade': 0.02,
            'max_positions': 5,
            'strategies': ['multi_signal'],
            'alert_thresholds': {
                'daily_loss_limit': 0.05,
                'position_loss_limit': 0.03,
                'profit_target': 0.08
            },
            'update_interval_minutes': 15
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}. Using defaults.")
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default config file: {config_file}")
        
        return default_config
    
    def _initialize_strategies(self) -> Dict:
        strategies = {}
        for strategy_name in self.config.get('strategies', ['multi_signal']):
            if strategy_name == 'multi_signal':
                strategies['multi_signal'] = MultiSignalStrategy()
            elif strategy_name == 'momentum':
                strategies['momentum'] = MomentumStrategy()
            elif strategy_name == 'mean_reversion':
                strategies['mean_reversion'] = MeanReversionStrategy()
        
        return strategies
    
    def _is_market_hours(self) -> bool:
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        
        if now.weekday() >= 5:  # Weekend
            return False
        
        start_time = self.config['trading_hours']['start']
        end_time = self.config['trading_hours']['end']
        
        return start_time <= current_time <= end_time
    
    def _check_daily_loss_limit(self) -> bool:
        daily_pnl = self._calculate_daily_pnl()
        daily_loss_pct = daily_pnl / self.trader.initial_capital
        
        if daily_loss_pct <= -self.alert_thresholds['daily_loss_limit']:
            self.logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2%}")
            self._send_alert(f"ALERT: Daily loss limit reached: {daily_loss_pct:.2%}")
            return True
        return False
    
    def _calculate_daily_pnl(self) -> float:
        today = datetime.now().date()
        daily_trades = [t for t in self.trader.trade_history 
                       if t.get('timestamp', datetime.now()).date() == today and 'pnl' in t]
        return sum(t['pnl'] for t in daily_trades)
    
    def _send_alert(self, message: str):
        self.logger.warning(f"ALERT: {message}")
        # Here you could integrate with email, SMS, Slack, etc.
        # For now, just log the alert
    
    def _check_position_alerts(self):
        for symbol, position in self.trader.positions.items():
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price
            
            if pnl_pct <= -self.alert_thresholds['position_loss_limit']:
                self._send_alert(f"{symbol} position down {pnl_pct:.2%}")
            elif pnl_pct >= self.alert_thresholds['profit_target']:
                self._send_alert(f"{symbol} position up {pnl_pct:.2%} - consider taking profit")
    
    def update_market_data(self):
        self.logger.info("Updating market data and positions...")
        try:
            self.trader.update_positions(self.watchlist)
            self._check_position_alerts()
            self.last_update = datetime.now()
            
            current_portfolio_value = self.trader.get_portfolio_value()
            self.performance_log.append({
                'timestamp': self.last_update,
                'portfolio_value': current_portfolio_value,
                'positions': len(self.trader.positions),
                'cash': self.trader.current_capital
            })
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def scan_for_signals(self):
        if not self._is_market_hours():
            self.logger.info("Market is closed. Skipping signal scan.")
            return
        
        if self._check_daily_loss_limit():
            self.logger.warning("Daily loss limit reached. Stopping trading.")
            self.trading_active = False
            return
        
        self.logger.info("Scanning for trading signals...")
        
        for symbol in self.watchlist:
            try:
                if len(self.trader.positions) >= self.config.get('max_positions', 5):
                    self.logger.info("Maximum positions reached. Skipping new entries.")
                    break
                
                for strategy_name, strategy in self.active_strategies.items():
                    signal = self.trader.analyze_stock(symbol, 'macd')  # Use MACD as base
                    
                    if strategy_name != 'macd':
                        custom_signal = strategy.generate_signal(self.trader.get_stock_data(symbol))
                        if custom_signal.confidence > signal.confidence:
                            signal = custom_signal
                    
                    if signal.confidence > 0.7:
                        self.logger.info(f"{symbol}: {signal.signal} signal (confidence: {signal.confidence:.2f}) - {signal.reason}")
                        
                        if self.trading_active:
                            self.trader.execute_trade(symbol, signal, self.config.get('risk_per_trade', 0.02))
                        else:
                            self.logger.info("Trading not active. Signal logged only.")
            
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
    
    def generate_daily_report(self):
        metrics = self.trader.get_performance_metrics()
        daily_pnl = self._calculate_daily_pnl()
        
        report = f"""
=== Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')} ===
Portfolio Value: ${metrics['portfolio_value']:,.2f}
Daily P&L: ${daily_pnl:,.2f}
Total Return: {metrics['total_return_pct']:.2f}%
Active Positions: {metrics['positions']}
Total Trades Today: {len([t for t in self.trader.trade_history if t.get('timestamp', datetime.now()).date() == datetime.now().date()])}
Win Rate: {metrics['win_rate_pct']:.2f}%
Available Cash: ${metrics['current_capital']:,.2f}

=== Current Positions ===
"""
        for symbol, position in self.trader.positions.items():
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price * 100
            report += f"{symbol}: {position.shares} shares @ ${position.entry_price:.2f} (Current: ${position.current_price:.2f}, P&L: {pnl_pct:+.2f}%)\n"
        
        self.logger.info(report)
        return report
    
    def start_trading(self):
        self.trading_active = True
        self.logger.info("Real-time trading started!")
        
        # Schedule market data updates
        update_interval = self.config.get('update_interval_minutes', 15)
        schedule.every(update_interval).minutes.do(self.update_market_data)
        
        # Schedule signal scanning
        schedule.every(5).minutes.do(self.scan_for_signals)
        
        # Schedule daily report
        schedule.every().day.at("16:30").do(self.generate_daily_report)
        
        self.logger.info(f"Scheduled updates every {update_interval} minutes")
        self.logger.info("Scheduled signal scanning every 5 minutes during market hours")
        self.logger.info("Scheduled daily report at 16:30")
        
        try:
            while self.trading_active:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user.")
        except Exception as e:
            self.logger.error(f"Trading error: {e}")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        self.trading_active = False
        self.logger.info("Real-time trading stopped.")
        final_report = self.generate_daily_report()
        
        # Save performance log
        performance_df = pd.DataFrame(self.performance_log)
        if not performance_df.empty:
            performance_df.to_csv(f"performance_log_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
            self.logger.info("Performance log saved.")
    
    def run_simulation_mode(self, duration_hours: int = 1):
        self.logger.info(f"Starting simulation mode for {duration_hours} hours...")
        self.trading_active = True
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time and self.trading_active:
            self.update_market_data()
            self.scan_for_signals()
            
            self.logger.info(f"Simulation running... {(end_time - datetime.now()).seconds // 60} minutes remaining")
            time.sleep(300)  # Wait 5 minutes between cycles
        
        self.stop_trading()
        self.logger.info("Simulation completed!")

def create_sample_config():
    config = {
        "initial_capital": 100000,
        "watchlist": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"],
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
    
    with open("trader_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: trader_config.json")

def main():
    print("=== Real-Time Stock Trader ===")
    print("1. Create sample configuration")
    print("2. Run simulation mode (1 hour)")
    print("3. Start real trading")
    print("4. Exit")
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        create_sample_config()
    elif choice == "2":
        trader = RealTimeTrader()
        trader.run_simulation_mode(1)
    elif choice == "3":
        trader = RealTimeTrader()
        print("WARNING: This will start real trading. Make sure you have reviewed your configuration.")
        confirm = input("Are you sure you want to start real trading? (yes/no): ").strip().lower()
        if confirm == "yes":
            trader.start_trading()
        else:
            print("Trading cancelled.")
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()