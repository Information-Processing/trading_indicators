from collections import deque
import websocket
import threading
import json
from dataclasses import dataclass


MAX_TRADE_HISTORY = 50000 


@dataclass
class Trade:
    price: float
    volume: float
    time: float
    is_buyer_maker: bool


class BinanceWSClient:
    def __init__(self):
        self.trades = deque(maxlen=MAX_TRADE_HISTORY)
        self.order_book = {"bids": [], "asks": []}
        self.last_price = 0.0
        self.depth_count = 0
        self.trade_count = 0
        URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"

        self.ws = websocket.WebSocketApp(
            URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        ) 

        self.lock = threading.Lock()

    def on_open(self, ws):
        print('Websocket Open')

    def on_message(self, ws, message):
        self._handle_message(message)

    def on_error(self, ws, err):
        print(err)

    def on_close(self, ws, code, reason):
        print('Ws closed')

    def run_ws(self):
        self.t = threading.Thread(target=self.ws.run_forever, daemon=True)
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
        timestamp = float(data["T"] / 1000.0) # Binance sends milliseconds, convert to seconds
        is_buyer_maker =  bool(data["m"])

        new_trade = Trade(price, volume, timestamp, is_buyer_maker)
        with self.lock:
            self.trades.append(new_trade)

        self.last_price = price 
        self.trade_count += 1


    def _process_order_book(self, data):
        asks  = [(float(price), float(qty)) for price, qty in data.get("asks", [])]
        bids = [(float(price), float(qty)) for price, qty in data.get("bids", [])]
        
        self.order_book = {
            "asks": asks,
            "bids": bids
        }
        self.depth_count += 1
        #difference = asks[0][1] - bids[0][1]
        #imballance = bids[0][1] /(bids[0][1] + asks[0][1])
        #totalby = bids[0][1] * bids[0][0]
        #totalsell = asks[0][1] * asks[0][0]
        #print(f"asks: {asks[0]}, at: {totalsell:.5f}, bids: {bids[0]}, bt: {totalby:.5f}, qty diff: {difference:.5f}, imbalance: {imballance:.5f}")
        #print(f"asks : {asks}")

 
    def volume_of_best(self, data):
        best_ask = [(float(), float(qyt))]



    def get_trades_since(self, cutoff):
        # can be optimised
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
        while 1:
            if len(self.trades) > 0:
                recent_trade = self.trades[-1]
                price = recent_trade.price
                vol = recent_trade.volume
                ts = recent_trade.time
                is_buyer= recent_trade.is_buyer_maker
                print(f"price = {price}, volume = {vol}, timestamp = {ts}, maker = {is_buyer}")



if '__main__' == __name__:
    wsc = BinanceWSClient()
    wsc.run_ws()
    wsc.print_data()

    




