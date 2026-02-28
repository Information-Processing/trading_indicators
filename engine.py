from binance_ws import BinanceWSClient
from calc_engine import CalculationEngine 
import time


class Engine:
    def __init__(self):
        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()

        self.ce = CalculationEngine()
        
    def get_data(self):
        # trades = self.binance_ws.trades
        # order_book = self.binance_ws.order_book
        # depth_count = self.binance_ws.depth_count
        # last_price = self.binance_ws.last_price
        # depth_count = self.binance_ws.depth_count
        # trade_count = self.binance_ws.trade_count
        while(1):
            now = time.time()
            
            trades_10 = self.binance_ws.get_trades_since(now - 11.0)
            trades_30 = self.binance_ws.get_trades_since(now - 31.0)
            trades_60 = self.binance_ws.get_trades_since(now - 61.0) 

            vwma_10 = self.ce.vwma_calculate(trades_10, now, 10.0)
            vwma_30 = self.ce.vwma_calculate(trades_30, now, 30.0)
            vwma_60 = self.ce.vwma_calculate(trades_60, now, 60.0) 

            time.sleep(1)

            print(f"VWMA [10 sec: {vwma_10}], [30 sec: {vwma_30}], [60 sec: {vwma_60}]")
            

        
        return {
                "vwma_10" : vwma_10
                }

if __name__ == '__main__':
    eng = Engine()
    eng.get_data()


