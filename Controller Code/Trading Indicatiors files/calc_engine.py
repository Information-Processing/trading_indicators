import numpy as np
from collections import deque


class CalculationEngineV2:
    def __init__(self, intensity_window=20):
        self.trade_count_history = deque(maxlen=intensity_window)
        self.vdelta_history = deque(maxlen=20)

    # ----------------------------------------------------------
    # 1. SPREAD (basis points)
    # ----------------------------------------------------------

    def spread_bps(self, best_ask_price, best_bid_price):
        mid = (best_ask_price + best_bid_price) / 2.0
        if mid == 0:
            return 0.0
        return (best_ask_price - best_bid_price) / mid * 10_000

    # ----------------------------------------------------------
    # 2. WEIGHTED MID-PRICE DEVIATION (micro-price signal)
    # ----------------------------------------------------------

    def weighted_mid_deviation(self, best_ask, best_bid):
        ask_p, ask_q = best_ask
        bid_p, bid_q = best_bid
        mid = (ask_p + bid_p) / 2.0
        denom = bid_q + ask_q
        if denom == 0 or mid == 0:
            return 0.0
        micro = bid_p * ask_q + ask_p * bid_q
        micro /= denom
        return (micro - mid) / mid * 10_000

    # ----------------------------------------------------------
    # 3. MULTI-LEVEL BOOK IMBALANCE  (-1 to +1)
    # ----------------------------------------------------------

    def book_imbalance(self, asks, bids, levels=None):
        if levels is not None:
            asks = asks[:levels]
            bids = bids[:levels]
        bid_qty = sum(q for _, q in bids)
        ask_qty = sum(q for _, q in asks)
        denom = bid_qty + ask_qty
        if denom == 0:
            return 0.0
        return (bid_qty - ask_qty) / denom

    # ----------------------------------------------------------
    # 4. ORDER BOOK SLOPE RATIO (log-ratio, ~centred on 0)
    # ----------------------------------------------------------

    def book_slope_ratio(self, asks, bids, levels=5):
        bid_slope = self._cum_qty_slope(bids[:levels])
        ask_slope = self._cum_qty_slope(asks[:levels])
        if ask_slope == 0:
            return 0.0
        ratio = bid_slope / ask_slope if ask_slope != 0 else 1.0
        if ratio <= 0:
            return 0.0
        return float(np.log(ratio))

    @staticmethod
    def _cum_qty_slope(levels):
        if len(levels) < 2:
            return 0.0
        best_price = levels[0][0]
        dists = np.array([abs(p - best_price) for p, _ in levels])
        cum_qty = np.cumsum([q for _, q in levels])
        if dists[-1] == 0:
            return 0.0
        x = dists - dists.mean()
        y = cum_qty - cum_qty.mean()
        denom = (x * x).sum()
        if denom == 0:
            return 0.0
        return float((x * y).sum() / denom)

    # ----------------------------------------------------------
    # 5. DEPTH RATIO  (log-ratio, ~centred on 0)
    # ----------------------------------------------------------

    def depth_ratio(self, asks, bids):
        bid_d = sum(q for _, q in bids)
        ask_d = sum(q for _, q in asks)
        if ask_d == 0 or bid_d == 0:
            return 0.0
        return float(np.log(bid_d / ask_d))

    # ----------------------------------------------------------
    # 6. VOLUME DELTA RATIO  (-1 to +1)
    # ----------------------------------------------------------

    def volume_delta_ratio(self, trades, now, window_secs):
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if not recent:
            return 0.0
        buy_vol = sum(t.price * t.volume for t in recent if not t.is_buyer_maker)
        sell_vol = sum(t.price * t.volume for t in recent if t.is_buyer_maker)
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    # ----------------------------------------------------------
    # 7. TRADE INTENSITY Z-SCORE
    # ----------------------------------------------------------

    def trade_intensity_zscore(self, trade_count_in_window):
        self.trade_count_history.append(trade_count_in_window)
        if len(self.trade_count_history) < 3:
            return 0.0
        arr = np.array(self.trade_count_history)
        mu = arr.mean()
        std = arr.std()
        if std < 1e-9:
            return 0.0
        return float((trade_count_in_window - mu) / std)

    # ----------------------------------------------------------
    # 8. LARGE TRADE RATIO  (0 to 1)
    # ----------------------------------------------------------

    def large_trade_ratio(self, trades, now, window_secs, threshold_qty=0.1):
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if not recent:
            return 0.0
        total_vol = sum(t.volume for t in recent)
        large_vol = sum(t.volume for t in recent if t.volume >= threshold_qty)
        if total_vol == 0:
            return 0.0
        return large_vol / total_vol

    # ----------------------------------------------------------
    # 9. REALISED VOLATILITY (annualised-ish, but stable scale)
    # ----------------------------------------------------------

    def realized_volatility(self, trades, now, window_secs):
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 2:
            return 0.0
        prices = np.array([t.price for t in recent])
        log_ret = np.diff(np.log(prices))
        if len(log_ret) == 0:
            return 0.0
        return float(log_ret.std() * 10_000)

    # ----------------------------------------------------------
    # 10. MOMENTUM (basis points return over window)
    # ----------------------------------------------------------

    def momentum(self, trades, now, window_secs):
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 2:
            return 0.0
        p_old = recent[0].price
        p_new = recent[-1].price
        if p_old == 0:
            return 0.0
        return (p_new - p_old) / p_old * 10_000

    # ----------------------------------------------------------
    # 11. VWMA DEVIATION (basis points from VWMA)
    # ----------------------------------------------------------

    def vwma_deviation(self, trades, now, window_secs, last_price):
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if not recent:
            return 0.0
        prices = np.array([t.price for t in recent])
        volumes = np.array([t.volume for t in recent])
        total_vol = volumes.sum()
        if total_vol == 0:
            return 0.0
        vwma = float((prices * volumes).sum() / total_vol)
        if vwma == 0:
            return 0.0
        return (last_price - vwma) / vwma * 10_000

    # ----------------------------------------------------------
    # 12. CUMULATIVE VOLUME DELTA (rolling sum of delta ratio)
    # ----------------------------------------------------------

    def cumulative_volume_delta(self, current_vdelta):
        self.vdelta_history.append(current_vdelta)
        return float(sum(self.vdelta_history))
