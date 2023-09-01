import numpy as np
import yfinance as yf

# CODE SOURCE: https://blankly.finance/build-a-backtester/
# Using for educational purposes

# yfinance gets data from yahoo finance

# Ticker for sp500 for period of 2yrs. (See Vectoriser)
spy = yf.Ticker('SPY')
hist = spy.history(period='2y')
pers = hist.loc[:, 'Close'].apply(np.log).diff(1)

# We convert our data to NumPy for easy slicing and calculate
# our running SMAs for a start to trading on day 200.
prices = hist.loc[:, 'Close'].to_numpy()
running_sma50 = prices[150:200].mean()
running_sma200 = prices[:200].mean()

# Like before, we track the operation to do on a given day — buy (1), short sell (-1), or stay neutral (0),
# and we store it in the variable event. Our array rets tracks log returns. Each day we have data for (from 200 til
# the end), we first take the action calculated by the signal on the previous day, execute it, and append to rets.
# Then, we update the running SMAs by adding the most recent price point, subtracting the one that should be dropped
# (from 50/200 days ago), and dividing that difference by the appropriate period. Finally, we calculate the
# appropriate operation by taking the sign of the difference between SMAs.
event = 0
rets = [0]
for i in range(200, hist.shape[0]):
    rets.append(rets[-1] + (event * (pers.iloc[i])))
    running_sma50 += (prices[i] - prices[i - 50]) / 50
    running_sma200 += (prices[i] - prices[i - 200]) / 200
    event = np.sign(running_sma50 - running_sma200)

# Once we exponentiate, we’re left with the cumulative returns. Like before, we get the 1.19 value, corresponding to
# an ~ 19% gain. While the vectorized backtesting may have been faster, we had sacrificed flexibility. Real trading
# algorithms often have to dynamically allocate capital, something that vectorized backtesting often doesn’t allow.
# Although an event-based backtester does address some of these concerns, it is far from perfect.
print(np.exp(rets))
