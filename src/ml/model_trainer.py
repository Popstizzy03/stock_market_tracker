
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

from src.core.trader import TechnicalIndicators

def create_features(data: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """Create features for the machine learning model."""
    df = data.copy()

    # Technical Indicators
    indicators = TechnicalIndicators()
    df['SMA_10'] = indicators.sma(df['Close'], 10)
    df['SMA_30'] = indicators.sma(df['Close'], 30)
    df['EMA_10'] = indicators.ema(df['Close'], 10)
    df['EMA_30'] = indicators.ema(df['Close'], 30)
    df['RSI'] = indicators.rsi(df['Close'])
    df['ATR'] = indicators.atr(df['High'], df['Low'], df['Close'])

    macd_line, signal_line, _ = indicators.macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line

    # Lag Features
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Return_Lag_{i}'] = df['Close'].pct_change().shift(i)

    # Target Variable
    df['Future_Return'] = df['Close'].pct_change().shift(-5)
    df['Target'] = (df['Future_Return'] > 0.01).astype(int)  # 1 for BUY, 0 for HOLD/SELL

    return df.dropna()

def train_model(symbol: str = 'AAPL', start_date: str = '2020-01-01', end_date: str = '2024-01-01'):
    """Trains a RandomForestClassifier and saves it to a file."""
    print(f"Fetching data for {symbol}...")
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data for {symbol}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print("Creating features...")
    featured_data = create_features(stock_data)

    if featured_data.empty:
        print("Not enough data to create features.")
        return

    features = [col for col in featured_data.columns if col not in ['Target', 'Future_Return']]
    X = featured_data[features]
    y = featured_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Ensure the ml directory exists
    if not os.path.exists('src/ml'):
        os.makedirs('src/ml')

    model_path = 'src/ml/trading_model.joblib'
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
