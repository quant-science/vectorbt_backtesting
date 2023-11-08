# QUANT SCIENCE UNIVERSITY
# Goal: Get you started making progress with algorithmic trading
# Pro-Tip: Visualize the Moving Average Crossover Strategy (At the End)
# ****

# Libraries
import vectorbt as vbt
import pandas as pd
import numpy as np

# Read data
price_aapl = pd.read_pickle("data/price_aapl.pkl")
price_aapl

# Profit Level: Buy and Hold Strategy
pf_buy_hold = vbt.Portfolio.from_holding(
    close=price_aapl, 
    init_cash=10_000
)
pf_buy_hold.total_profit()


# 1.0 Simple Moving Average Crossover Strategy 5-20 Day (AAPL)

# 1. Define the strategy
fast_ma = vbt.MA.run(price_aapl, window=5)
slow_ma = vbt.MA.run(price_aapl, window=20)

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# 2. Run the strategy
pf_ma_strat = vbt.Portfolio.from_signals(
    close = price_aapl, 
    entries=entries, 
    exits=exits, 
    init_cash=10_000,
    fees=0.001,
)

pf_ma_strat.total_profit()

pf_ma_strat.stats()

pf_ma_strat.plot().show()

# 2.0 Backtesting 100 windows for the Moving Average Crossover Strategy (AAPL) 

windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(
    close = price_aapl, 
    window=windows, 
    r=2, 
    short_names=['fast', 'slow']
)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf_100_ma_strats = vbt.Portfolio.from_signals(
    close = price_aapl, 
    entries = entries, 
    exits = exits, 
    size=np.inf, 
    fees=0.001, 
    freq='1D',
    init_cash=10_000
)

pf_100_ma_strats.total_profit().max()
pf_100_ma_strats.total_profit().idxmax()

best_index = pf_100_ma_strats.total_profit().idxmax()

stats = pf_100_ma_strats[best_index].stats()
stats

pf_100_ma_strats[best_index].plot().show()

# Conclusions ----
# You can do this!
# There's a lot more to learn:
# - More Trading Strategies
# - Risk Management
# - Portfolio Optimization
# - Machine Learning
# - Advanced Backtesting
# - Live Trading & Execution
