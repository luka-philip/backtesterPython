import numpy as np
import yfinance as yf

# CODE SOURCE: https://blankly.finance/build-a-backtester/
# Using for educational purposes

# yfinance gets data from yahoo finance
# Ticker for sp500 for period of 2yrs We take the log of prices so that we can add
# differentials from day to day and then exponentiate to get the overall return.
# Diff(1) takes the difference between each pair of elements, allowing us to consider
# the day-to-day returns, which can be easily combined to represent buying and selling at different times.
spy = yf.Ticker('SPY')
hist = spy.history(period='2y')
pers = hist.loc[:, 'Close'].apply(np.log).diff(1)


# returns a Pandas DataFrame that contains the specified moving average for each day.
def sma(prices, period):
    return prices.loc[:, 'Close'].rolling(window=period).mean()


# We define our signal — the golden cross. After calculating each moving average,
# we zero elements in the first 200 days, where the SMA200 isn’t defined. We then
# take the sign of each, so that our signal tells us to go long when SMA50 is over the
# SMA200 and go short when the SMA50 is under, staying out when the two are equal.
sma50 = sma(hist, 50).fillna(0).shift(periods=1, fill_value=0)
sma50[:200] = 0
sma200 = sma(hist, 200).fillna(0).shift(periods=1, fill_value=0)
signal = sma50 - sma200
sig = signal.apply(np.sign)

# We run the signal over the log-prices and exponentiate to get our final values of the backtest.
returns = sig * pers
print(returns.cumsum().apply(np.exp))
