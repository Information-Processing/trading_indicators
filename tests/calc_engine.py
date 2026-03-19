import numpy as np
from collections import deque


class CalculationEngineV2:
    def __init__(self, intensity_window=20):
        self.trade_count_history = deque(maxlen=intensity_window)
        self.vdelta_history = deque(maxlen=20)
        # State for new indicators
        self.trade_size_history = deque(maxlen=20)
        self.mid_history = deque(maxlen=50)
        self.interval_history = deque(maxlen=20)

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

    # ----------------------------------------------------------
    # 13. PRICE ACCELERATION (bps — short vs long momentum diff)
    # ----------------------------------------------------------

    def price_acceleration(self, trades, now, short_secs=5.0, long_secs=10.0):
        """Difference between short and long momentum in bps: proxy for d²p/dt²."""
        def _mom(window_secs):
            recent = [t for t in trades if t.time >= now - window_secs]
            if len(recent) < 2:
                return 0.0
            p_old, p_new = recent[0].price, recent[-1].price
            return 0.0 if p_old == 0 else (p_new - p_old) / p_old * 10_000

        return _mom(short_secs) - _mom(long_secs)

    # ----------------------------------------------------------
    # 14. BUY / SELL TRADE COUNT RATIO  (0 to 1)
    # ----------------------------------------------------------

    def buy_sell_trade_count_ratio(self, trades, now, window_secs=5.0):
        """Fraction of aggressive-buy trades (is_buyer_maker=False) over window."""
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if not recent:
            return 0.5
        buy_count = sum(1 for t in recent if not t.is_buyer_maker)
        return buy_count / len(recent)

    # ----------------------------------------------------------
    # 15. TRADE SIZE Z-SCORE
    # ----------------------------------------------------------

    def trade_size_zscore(self, trades, now, window_secs=5.0):
        """Z-score of current average trade size vs rolling window history."""
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        avg_size = float(np.mean([t.volume for t in recent])) if recent else 0.0
        self.trade_size_history.append(avg_size)
        if len(self.trade_size_history) < 3:
            return 0.0
        arr = np.array(self.trade_size_history)
        std = arr.std()
        if std < 1e-12:
            return 0.0
        return float((avg_size - arr.mean()) / std)

    # ----------------------------------------------------------
    # 16. PRICE RANGE (bps — high/low spread within window)
    # ----------------------------------------------------------

    def price_range_bps(self, trades, now, window_secs=10.0):
        """(high − low) / mid × 10 000 over the window."""
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 2:
            return 0.0
        prices = [t.price for t in recent]
        mid = (max(prices) + min(prices)) / 2.0
        if mid == 0:
            return 0.0
        return (max(prices) - min(prices)) / mid * 10_000

    # ----------------------------------------------------------
    # 17. TICK DIRECTION RATIO  (0 to 1)
    # ----------------------------------------------------------

    def tick_direction_ratio(self, trades, now, window_secs=5.0):
        """Fraction of consecutive up-ticks (price rose vs previous trade)."""
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 2:
            return 0.5
        upticks = sum(
            1 for i in range(1, len(recent)) if recent[i].price > recent[i - 1].price
        )
        return upticks / (len(recent) - 1)

    # ----------------------------------------------------------
    # 18. SHORT-WINDOW VWAP DEVIATION (bps)
    # ----------------------------------------------------------

    def vwap_deviation_short(self, trades, now, window_secs=5.0, last_price=None):
        """Deviation of last_price from VWAP over a short window, in bps."""
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if not recent or last_price is None:
            return 0.0
        prices = np.array([t.price for t in recent])
        volumes = np.array([t.volume for t in recent])
        total_vol = volumes.sum()
        if total_vol == 0:
            return 0.0
        vwap = float((prices * volumes).sum() / total_vol)
        if vwap == 0:
            return 0.0
        return (last_price - vwap) / vwap * 10_000

    # ----------------------------------------------------------
    # 19. ASK TOUCH PRESSURE  (0 to 1)
    # ----------------------------------------------------------

    def ask_touch_pressure(self, asks):
        """Fraction of total ask-side volume sitting at the best ask level."""
        if not asks:
            return 0.0
        total_qty = sum(q for _, q in asks)
        if total_qty == 0:
            return 0.0
        return asks[0][1] / total_qty

    # ----------------------------------------------------------
    # 20. BID TOUCH PRESSURE  (0 to 1)
    # ----------------------------------------------------------

    def bid_touch_pressure(self, bids):
        """Fraction of total bid-side volume sitting at the best bid level."""
        if not bids:
            return 0.0
        total_qty = sum(q for _, q in bids)
        if total_qty == 0:
            return 0.0
        return bids[0][1] / total_qty

    # ----------------------------------------------------------
    # 21. ROLL SPREAD ESTIMATE (bps — from price autocorrelation)
    # ----------------------------------------------------------

    def roll_spread_estimate(self, trades, now, window_secs=10.0):
        """
        Roll (1984) model: spread ≈ 2√(−cov(Δp_t, Δp_{t−1})).
        Negative autocovariance indicates bid-ask bounce; returned in bps.
        """
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 3:
            return 0.0
        prices = np.array([t.price for t in recent])
        dp = np.diff(prices)
        if len(dp) < 2:
            return 0.0
        cov = float(np.cov(dp[:-1], dp[1:])[0, 1])
        if cov >= 0:
            return 0.0
        mid = prices.mean()
        if mid == 0:
            return 0.0
        spread_price = 2.0 * np.sqrt(-cov)
        return spread_price / mid * 10_000

    # ----------------------------------------------------------
    # 22. MID-PRICE RETURNS Z-SCORE
    # ----------------------------------------------------------

    def mid_price_returns_zscore(self, asks, bids):
        """
        Z-score of the current mid-price log-return vs rolling mid history.
        Captures mean-reversion / trending conditions in the book mid.
        """
        if not asks or not bids:
            return 0.0
        mid = (asks[0][0] + bids[0][0]) / 2.0
        self.mid_history.append(mid)
        if len(self.mid_history) < 3:
            return 0.0
        mids = np.array(self.mid_history)
        log_rets = np.diff(np.log(mids[mids > 0]))
        if len(log_rets) < 2:
            return 0.0
        std = log_rets.std()
        if std < 1e-12:
            return 0.0
        return float(log_rets[-1] / std)

    # ----------------------------------------------------------
    # 23. TRADE SIZE DISPERSION (coefficient of variation)
    # ----------------------------------------------------------

    def trade_size_dispersion(self, trades, now, window_secs=5.0):
        """
        Coefficient of variation (std / mean) of trade sizes in window.
        High values indicate heterogeneous (potentially informed) order flow.
        """
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 2:
            return 0.0
        sizes = np.array([t.volume for t in recent])
        mean_s = sizes.mean()
        if mean_s < 1e-12:
            return 0.0
        return float(sizes.std() / mean_s)

    # ----------------------------------------------------------
    # 24. INTER-TRADE INTERVAL Z-SCORE
    # ----------------------------------------------------------

    def inter_trade_interval_zscore(self, trades, now, window_secs=5.0):
        """
        Z-score of current average inter-trade interval vs rolling history.
        Negative → trading faster than usual (high urgency / activity).
        """
        cutoff = now - window_secs
        recent = [t for t in trades if t.time >= cutoff]
        if len(recent) < 2:
            avg_interval = 0.0
        else:
            times = [t.time for t in recent]
            avg_interval = float(np.mean(np.diff(times)))
        self.interval_history.append(avg_interval)
        if len(self.interval_history) < 3:
            return 0.0
        arr = np.array(self.interval_history)
        std = arr.std()
        if std < 1e-12:
            return 0.0
        return float((avg_interval - arr.mean()) / std)
