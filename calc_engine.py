import numpy as np
from collections import deque


class CalculationEngine:
    def __init__(self):
        print("calculation engine initialised")
        self.bid_depth_history = deque(maxlen=20)

    def vwma_calculate(self, trades, now, window_secs):
        # volume weighted moving avg
        cutoff = now - window_secs 
        recent_trades = [t for t in trades if t.time >= cutoff]

        prices = np.array([t.price for t in recent_trades])
        volumes = np.array([t.volume for t in recent_trades])
        total_volume = volumes.sum()
        weighted_prices_sum = (prices * volumes).sum()

        vwap = weighted_prices_sum / total_volume
        
        return float(vwap)

    def imbalance_calc(self, asks, bids):
        return bids[1] / (bids[1] + asks[1])

    def price_depth(self, levels):
        arr = np.array(levels)
        prices = arr[:, 0]
        qtys = arr[:, 1]
        return float(np.sum(prices * qtys))

    def topvolume(self, levels, cutoff):
        arr = np.array(levels)
        prices = arr[:cutoff, 0]
        qtys = arr[:cutoff, 1]
        return float(np.sum(prices * qtys))

    def bottomvolume(self, levels, cutoff):
        arr = np.array(levels)
        prices = arr[cutoff:, 0]
        qtys = arr[cutoff:, 1]
        return float(np.sum(prices * qtys))

    def dropoff(self, levels, cutoff):
        top = self.topvolume(levels, cutoff)
        bottom = self.bottomvolume(levels, cutoff)
        return bottom / top

    def depth_average(self, depth_value):
        self.bid_depth_history.append(depth_value)
        if len(self.bid_depth_history) == 0:
            return 0.0
        return float(np.mean(self.bid_depth_history))


    def sell_total(self, trades, now, window_secs):
        cutoff = now - window_secs 
        sellers = [t for t in trades if t.time >= cutoff and t.is_buyer_maker]
        if len(sellers) == 0:
            return 0
        prices = np.array([t.price for t in sellers])
        volumes = np.array([t.volume for t in sellers])
        return float(np.sum(prices * volumes))


    def bought_total(self, trades, now, window_secs):
        cutoff = now - window_secs 
        sellers = [t for t in trades if t.time >= cutoff and not t.is_buyer_maker]
        if len(sellers) == 0:
            return 0
        prices = np.array([t.price for t in sellers])
        volumes = np.array([t.volume for t in sellers])
        return float(np.sum(prices * volumes))

    