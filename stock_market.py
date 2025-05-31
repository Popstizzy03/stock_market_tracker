import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download stock data
ticker = "AAPL"  # Apple Inc.
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

ticker = ["AAPL", "MSFT"]
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")['Close']
data.plot(figsize=(12,6), title="AAPL vs MSFT Closing Prices")
plt.grid(True)
plt.show()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='AAPL Close Price', color='blue')
plt.plot(data['MA20'], label='20-Day MA', color='orange')
plt.plot(data['MA50'], label='50-Day MA', color='green')
plt.title('Apple (AAPL) Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
