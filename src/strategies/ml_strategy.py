
import pandas as pd
import joblib
from datetime import datetime
import os

from src.core.trader import TradingStrategy, TradingSignal, TechnicalIndicators
from src.ml.model_trainer import create_features

class MLTradingStrategy(TradingStrategy):
    def __init__(self, model_path: str = 'src/ml/trading_model.joblib'):
        super().__init__("Machine Learning Strategy")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)
        self.indicators = TechnicalIndicators()
        self.feature_cache = {}

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Generates a trading signal using the trained machine learning model.
        """
        if len(data) < 50:  # Minimum data required for feature creation
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Insufficient data")

        # Use the first timestamp as a cache key
        cache_key = data.index[0]

        if cache_key in self.feature_cache:
            featured_data = self.feature_cache[cache_key]
        else:
            # 1. Feature Engineering
            featured_data = create_features(data)
            self.feature_cache[cache_key] = featured_data

        if featured_data.empty:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), "Could not generate features")

        # 2. Prepare latest data for prediction
        latest_features = featured_data.iloc[-1:]
        features_for_prediction = [col for col in latest_features.columns if col not in ['Target', 'Future_Return']]
        X_latest = latest_features[features_for_prediction]

        # 3. Make Prediction
        try:
            prediction = self.model.predict(X_latest)[0]
            probability = self.model.predict_proba(X_latest)[0]
        except Exception as e:
            return TradingSignal("HOLD", 0.0, data['Close'].iloc[-1], datetime.now(), f"Prediction error: {e}")

        current_price = data['Close'].iloc[-1]

        # 4. Generate Signal
        if prediction == 1:  # BUY signal
            confidence = probability[1]  # Probability of the 'BUY' class
            return TradingSignal("BUY", confidence, current_price, datetime.now(), "ML model BUY signal")
        else:  # HOLD/SELL signal (model predicts 0)
            # For now, we'll treat 0 as a trigger to sell if a position is open
            confidence = probability[0] # Probability of the 'HOLD/SELL' class
            return TradingSignal("SELL", confidence, current_price, datetime.now(), "ML model SELL/HOLD signal")
