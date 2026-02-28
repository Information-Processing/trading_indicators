import numpy as np


class CalculationEngine:
    def __init__(self):
        print("calculation engine initialised")

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
      

