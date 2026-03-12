from collections import deque
import websocket
import threading
import json
import time
from dataclasses import dataclass

MAX_TRADE_HISTORY = 50000

@dataclass
class Trade:
    price: float
    volume: float
    time: float
    is_buyer_maker: bool

class BinanceWSClient:
    DEFAULT_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"

    def __init__(self, url=None):
        self.url = url or self.DEFAULT_URL
        self._url_changed = threading.Event()
        self.ws = None
        self.trades = deque(maxlen=MAX_TRADE_HISTORY)
        self.order_book = {"bids": [], "asks": []}
        self.last_price = 0.0
        self.depth_count = 0
        self.trade_count = 0
        self.lock = threading.Lock()

    def on_open(self, ws):
        print('Websocket Open')

    def on_message(self, ws, message):
        self._handle_message(message)

    def on_error(self, ws, err):
        print(err)

    def on_close(self, ws, code, reason):
        print(f'Ws closed (code={code}, reason={reason})')

    def update_url(self, target):
        self.url = f"wss://stream.binance.com:9443/stream?streams={target}usdt@trade/{target}usdt@depth20@100ms"
        self._url_changed.set()
        if self.ws:
            self.ws.close()

    def _run_forever_with_reconnect(self):
        while True:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
            if self._url_changed.is_set():
                self._url_changed.clear()
                print(f"URL changed, reconnecting immediately...")
            else:
                print("WebSocket disconnected, reconnecting in 3s...")
                time.sleep(3)

    def run_ws(self):
        self.t = threading.Thread(target=self._run_forever_with_reconnect, daemon=True)
        self.t.start()

    def _handle_message(self, raw_message):
        message = json.loads(raw_message)
        stream_name = message.get("stream", "")
        data = message.get("data", {})
        if "trade" in stream_name:
            self._process_trades(data)
        if "depth" in stream_name:
            self._process_order_book(data)

    def _process_trades(self, data):
        price = float(data["p"])
        volume = float(data["q"])
        timestamp = float(data["T"]) / 1000.0
        is_buyer_maker = bool(data["m"])
        new_trade = Trade(price, volume, timestamp, is_buyer_maker)
        with self.lock:
            self.trades.append(new_trade)
        self.last_price = price
        self.trade_count += 1

    def _process_order_book(self, data):
        asks = [(float(price), float(qty)) for price, qty in data.get("asks", [])]
        bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])]
        with self.lock:
            self.order_book = {
                "asks": asks,
                "bids": bids
            }
        self.depth_count += 1

    def get_trades_since(self, cutoff):
        with self.lock:
            trades_snapshot = list(self.trades)
        result = []
        for trade in reversed(trades_snapshot):
            if trade.time < cutoff:
                break
            result.append(trade)
        result.reverse()
        return result

    def print_data(self):
        while True:
            if len(self.trades) > 0:
                recent_trade = self.trades[-1]
                print(f"price = {recent_trade.price}, volume = {recent_trade.volume}, "
                      f"timestamp = {recent_trade.time}, maker = {recent_trade.is_buyer_maker}")

if __name__ == '__main__':
    wsc = BinanceWSClient()
    wsc.run_ws()
    wsc.print_data()
