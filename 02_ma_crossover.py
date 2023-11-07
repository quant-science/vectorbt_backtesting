

import vectorbt as vbt
import pandas as pd
import numpy as np

price_aapl = pd.read_pickle("data/price_aapl.pkl")

# Profit Level: Buy and Hold Strategy
pf = vbt.Portfolio.from_holding(price_aapl, init_cash=1000)
pf.total_profit()


# Profit Level: Moving Average Crossover Strategy (AAPL)

# 1. Define the strategy
fast_ma = vbt.MA.run(price_aapl, window=10)
slow_ma = vbt.MA.run(price_aapl, window=50)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# 2. Run the strategy
pf = vbt.Portfolio.from_signals(price_aapl, entries, exits, init_cash=1000)
pf.total_profit()


# Backtesting 100 windows for the Moving Average Crossover Strategy (AAPL) 

windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price_aapl, window=windows, r=2, short_names=['fast', 'slow'])
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(
    close = price_aapl, 
    entries = entries, 
    exits = exits, 
    size=np.inf, 
    fees=0.001, 
    freq='1D',
    init_cash=1000
)

pf.total_profit().max()
pf.total_profit().idxmax()

pf.total_profit().plot()

best_index = pf.total_profit().idxmax()

stats = pf[best_index].stats()
stats

pf[best_index].plot().show()


