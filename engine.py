from binance_ws import BinanceWSClient
from calc_engine import CalculationEngine 
import time
from collections import deque
import numpy as np


class Engine:
    def __init__(self):
        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()
        
        self.ce = CalculationEngine()
        self.netvoldelta = deque(maxlen = 10)
        
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

            vwma_10 = self.ce.vwma_calculate(trades, now, 10.0)
            
            shortterm_trades = self.binance_ws.get_trades_since(now - 1)
            total_sell = self.ce.sell_total(shortterm_trades, now, 1)
            total_brought =  self.ce.bought_total(shortterm_trades,now,1)
            self.netvoldelta.append(total_brought - total_sell)

            #Large trade volume ratio Average trade siz Trade arrival rate
            
            stlt = [t for t in shortterm_trades if t.volume > 0.1]
            stv = stlt = [t.volume for t in shortterm_trades]
            trade_arrival_rate =  len(shortterm_trades)
            average_Trade = sum(stv) / trade_arrival_rate if trade_arrival_rate > 0 else 1
            print(f"av {average_Trade:.5f},\t arrival rate {trade_arrival_rate},\t large trade volume, {sum(stlt):.5f}")
            #print(f"total sold: {total_sell}, total bought: {total_brought}, net delta:{sum(self.netvoldelta)}")
            
            
            # with this can do by/sell volume ratio and net volume 

            order_book = self.binance_ws.order_book
            asks = order_book.get("asks", "")
            bids = order_book.get("bids", "")
            if asks == []:
                time.sleep(1)
                continue
            else: 
                topasks = asks[0]
                topbids = bids[0]
                
                
            
            time.sleep (1)
            imballance = self.ce.imbalance_calc(topasks, topbids)
            askstotal=  self.ce.price_depth(asks)
            bidstotal = self.ce.price_depth(bids)
            ask_dropoff = self.ce.dropoff(asks, 5)
            bid_dropoff = self.ce.dropoff(bids, 5)

            ask_spread = self.ce.dropoff(asks, 3)

            #bid_depthroc = self.ce.bid_depth_roc(bids)

            #print(f"b drop{bid_dropoff:.7}, a drop{ask_dropoff:.7}, b tot {bidstotal:.7}, a tot{askstotal:.7}")
            #print(vwma_10)


        

if __name__ == '__main__':
    eng = Engine()
    eng.get_data()
    eng = Engine()


