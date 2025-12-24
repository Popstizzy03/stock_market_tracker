
from setuptools import setup, find_packages

setup(
    name='smart_stock_trader',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'yfinance>=0.2.18',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'schedule>=1.2.0',
        'scikit-learn>=1.3.0',
        'plotly>=5.0.0',
        'rich>=13.0.0',
    ],
    entry_points={
        'console_scripts': [
            'run_backtest=testing.backtesting:run_comprehensive_backtest',
            'run_realtime=real_time.real_time_trader:main',
            'train_model=ml.model_trainer:train_model',
        ],
    },
)
