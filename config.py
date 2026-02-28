#Binance ws
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"
SYMBOL = "btcusdt"
TRADE_STREAM = f"{SYMBOL}@trade" # real-time individual trades
DEPTH_STREAM = f"{SYMBOL}@depth20@100ms" # top 20 price levels, updated every 100ms

#Moving averages
MA_WINDOWS = [1.0, 3.0, 5.0] # standard: 1s, 3s, 5s
SHORT_MA_WINDOWS = [0.1, 0.5, 1.0] # fast/scalping: 100ms, 500ms, 1s

#Relative strength index
# Measures if BTC is being overbought (>70) or oversold (<30)
RSI_WINDOW = 14.0  # seconds of trade data to look at

#volatility and standard deviation
VOLATILITY_WINDOW = 10.0 # for log-return volatility
STD_DEV_WINDOW = 5.0 # for price standard deviation
SHORT_STD_DEV_WINDOW = 1.0 # fast price std dev

#volume
VOLUME_WINDOW = 10.0 # total BTC traded in the last 10 seconds

#memory
MAX_TRADE_HISTORY = 50000 # max trades to keep in memory
