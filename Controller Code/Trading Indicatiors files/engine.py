from binance_ws import BinanceWSClient
from calc_engine import CalculationEngineV2
import time
from collections import defaultdict
import threading


class Engine:
    def __init__(self):
        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()
        self.ce = CalculationEngineV2()
        self.ret_dict = defaultdict(list)

    def get_data(self):
        while True:
            now = time.time()
            order_book = self.binance_ws.order_book
            asks = order_book.get("asks", [])
            bids = order_book.get("bids", [])
            if not asks or not bids:
                time.sleep(1)
                continue
            trades_snapshot = list(self.binance_ws.trades)
            shortterm = self.binance_ws.get_trades_since(now - 1)
            best_ask, best_bid = asks[0], bids[0]
            last_price = self.binance_ws.last_price or (best_ask[0] + best_bid[0]) / 2
            vdelta = self.ce.volume_delta_ratio(trades_snapshot, now, 1.0)
            self.ret_dict["spread_bps"].append(self.ce.spread_bps(best_ask[0], best_bid[0]))
            self.ret_dict["weighted_mid_dev"].append(self.ce.weighted_mid_deviation(best_ask, best_bid))
            self.ret_dict["book_imbalance"].append(self.ce.book_imbalance(asks, bids))
            self.ret_dict["book_slope_ratio"].append(self.ce.book_slope_ratio(asks, bids, levels=5))
            self.ret_dict["depth_ratio"].append(self.ce.depth_ratio(asks, bids))
            self.ret_dict["vol_delta_ratio"].append(vdelta)
            self.ret_dict["trade_intensity_z"].append(self.ce.trade_intensity_zscore(len(shortterm)))
            self.ret_dict["large_trade_ratio"].append(self.ce.large_trade_ratio(trades_snapshot, now, 1.0))
            self.ret_dict["realized_vol"].append(self.ce.realized_volatility(trades_snapshot, now, 10.0))
            self.ret_dict["momentum"].append(self.ce.momentum(trades_snapshot, now, 10.0))
            self.ret_dict["vwma_deviation"].append(self.ce.vwma_deviation(trades_snapshot, now, 10.0, last_price))
            self.ret_dict["cum_vol_delta"].append(self.ce.cumulative_volume_delta(vdelta))
            self.ret_dict["last_price"].append(last_price)
            print(self.ret_dict)
            time.sleep(1)


if __name__ == '__main__':
    eng = Engine()
    threading.Thread(target=eng.get_data, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


