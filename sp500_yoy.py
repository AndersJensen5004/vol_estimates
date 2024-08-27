import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sp500 = yf.download('^GSPC', start='1928-01-01', end='2024-01-01', progress=False)

print(sp500.index.min(), sp500.index.max())

annual_returns = sp500['Adj Close'].resample('Y').ffill().pct_change() * 100

average_return = annual_returns.mean()
volatility = annual_returns.std()
positive_years = annual_returns[annual_returns > 0]
negative_years = annual_returns[annual_returns < 0]

positive_avg = positive_years.mean()
negative_avg = negative_years.mean()

print(f"Average Return: {average_return:.2f}%")
print(f"Volatility (Standard Deviation): {volatility:.2f}%")
print(f"Average Positive Year Return: {positive_avg:.2f}%")
print(f"Average Negative Year Return: {negative_avg:.2f}%")
print(f"Percentage of Positive Years: {len(positive_years) / len(annual_returns) * 100:.2f}%")

sns.set(style="whitegrid")

plt.figure(figsize=(16, 9))
bars = annual_returns.plot(kind='bar', color=['#ff4d4d' if r < 0 else '#33cc33' for r in annual_returns])

plt.axhline(average_return, color='#1f77b4', linestyle='--', linewidth=2, label=f'Average Return ({average_return:.2f}%)')
plt.title('S&P 500 Annual Returns (1928-2024)', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=16, fontweight='bold')
plt.ylabel('Annual Return (%)', fontsize=16, fontweight='bold')
plt.legend(fontsize=14)

bars.set_xticklabels([str(int(label.get_text()[:4])) for label in bars.get_xticklabels()], rotation=45, ha="right")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))  # Adjust to show fewer labels

plt.grid(visible=True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()
