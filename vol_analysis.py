import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

sp500 = yf.download('^GSPC', start='1928-01-01', end='2024-01-01', progress=False)

sp500['Daily Return'] = sp500['Adj Close'].pct_change()

sp500['1Y Volatility'] = sp500['Daily Return'].rolling(window=252).std() * np.sqrt(252) * 100
sp500['5Y Volatility'] = sp500['Daily Return'].rolling(window=252*5).std() * np.sqrt(252) * 100
sp500['10Y Volatility'] = sp500['Daily Return'].rolling(window=252*10).std() * np.sqrt(252) * 100

plt.figure(figsize=(14, 8))
plt.plot(sp500['1Y Volatility'], label='1-Year Volatility', color='blue', alpha=0.7)
plt.plot(sp500['5Y Volatility'], label='5-Year Volatility', color='orange', alpha=0.7)
plt.plot(sp500['10Y Volatility'], label='10-Year Volatility', color='green', alpha=0.7)
plt.axhline(sp500['1Y Volatility'].mean(), color='blue', linestyle='--', linewidth=2, label='1-Year Avg Volatility')
plt.axhline(sp500['5Y Volatility'].mean(), color='orange', linestyle='--', linewidth=2, label='5-Year Avg Volatility')
plt.axhline(sp500['10Y Volatility'].mean(), color='green', linestyle='--', linewidth=2, label='10-Year Avg Volatility')

plt.title('Historical Volatility of S&P 500 (1928-2024)', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=16)
plt.ylabel('Annualized Volatility (%)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

vix = yf.download('^VIX', start='1990-01-01', end='2024-01-01', progress=False)

sp500['Realized Volatility'] = sp500['Daily Return'].rolling(window=252).std() * np.sqrt(252) * 100

vix = vix['Close'].resample('M').last()
realized_vol = sp500['Realized Volatility'].resample('M').last()
combined = pd.DataFrame({'VIX (Implied Volatility)': vix, 'Realized Volatility': realized_vol}).dropna()

plt.figure(figsize=(14, 8))
plt.plot(combined['VIX (Implied Volatility)'], label='VIX (Implied Volatility)', color='red', linewidth=2)
plt.plot(combined['Realized Volatility'], label='Realized Volatility', color='blue', linewidth=2)
plt.title('Implied vs. Realized Volatility (1990-2024)', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=16)
plt.ylabel('Volatility (%)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

sp500['Cumulative Max'] = sp500['Adj Close'].cummax()
sp500['Drawdown'] = ((sp500['Adj Close'] / sp500['Cumulative Max']) - 1) * 100

plt.figure(figsize=(14, 8))
plt.plot(sp500['Drawdown'], label='Drawdown', color='purple', linewidth=2)
plt.title('S&P 500 Drawdowns (1928-2024)', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=16)
plt.ylabel('Drawdown (%)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()

drawdown_stats = sp500['Drawdown'].describe()
print(drawdown_stats)
