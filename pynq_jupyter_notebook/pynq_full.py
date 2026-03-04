from binance_ws import BinanceWSClient
from calc_engine import CalculationEngine 
from pynq import allocate
import time
from collections import deque, defaultdict
import numpy as np
import threading

# 1. DEFINE FEATURES - Exactly 12 features + 1 target = 13 total columns
FEATURE_RANGES = {
    "imballance" : 0,
    "asks_total" : 0,
    "bids_total" : 0,
    "ask_dropoff" : 0,
    "bid_dropoff" : 0,
    "ask_spread" : 0,
    "vwma_10" : 0,
    "vwma_5" : 0,
    "total_sell": 0,
    "total_brought" : 0,
    "stv" : 0,
    "trade_arrival_rate" : 0,  # 13th Column (The Target Y)
    "last_price": 0          # 12th Feature
}

# 2. ROBUST DATA PACKING
def bundle_dict_to_numpy(data_dict):
    """
    Converts ret_dict into a 2D numpy array. 
    Guarantees order based on FEATURE_RANGES.
    """
    keys = list(FEATURE_RANGES.keys())
    # Ensure all lists in dict have the same length by taking a snapshot
    min_len = min(len(data_dict[k]) for k in keys)
    cols = [data_dict[k][:min_len] for k in keys]
    return np.column_stack(cols)

# ---------------------------------------------------------
# LINEAR REGRESSION CLASSES (Corrected for D=13)
# ---------------------------------------------------------

class UnoptimisedSoftwareLR:
    def __init__(self, collumn_headers):
        self.collumn_headers = collumn_headers
        self.num_params = len(self.collumn_headers) # Should be 13
        self.a = np.empty((0, self.num_params))
        self.b = np.empty((0, 1))
        self.params = np.zeros((self.num_params, 1))

    def solve(self):
        ata = self.a.T @ self.a
        atb = self.a.T @ self.b
        # Use pinv to handle singular matrices during early data collection
        ata_inv = np.linalg.pinv(ata)
        return ata_inv @ atb

    def stream_chunk(self, lines):
        # lines has 13 columns. [:, :-1] takes 12 features.
        new_a = np.concatenate([lines[:, :-1], np.ones((len(lines), 1))], axis=1)
        new_b = lines[:, -1].reshape(-1, 1)
        self.a = np.vstack([self.a, new_a])
        self.b = np.vstack([self.b, new_b])
        self.params = self.solve()

    def print_equation(self, normaliser):
        denormed = normaliser.denormalise_weights(self.params)
        p = [v.item() for v in denormed.flatten()]
        print(f"{p[0]:.2f}*{self.collumn_headers[0]}", end="")
        for i in range(1, self.num_params - 1):
            print(f" + {p[i]:.2f}*{self.collumn_headers[i]}", end="")
        print(f" + {p[-1]:.2f}")

class OptimisedSoftwareLR:
    def __init__(self, collumn_headers): 
        self.collumn_headers = collumn_headers
        self.num_params = len(collumn_headers) # 13
        self.ata = np.zeros((self.num_params, self.num_params))
        self.atb = np.zeros((self.num_params, 1))
        self.params = np.zeros((self.num_params, 1))

    def stream_chunk_optimised(self, lines):
        line_output = lines[:, -1]
        q_mat = np.concatenate([lines[:,:-1], np.ones((lines.shape[0], 1))], axis=1)
        qt_mat = q_mat.transpose()
        self.ata += qt_mat @ q_mat
        self.atb += qt_mat @ line_output.reshape(-1, 1)
        self.recalculate_params()

    def recalculate_params(self):
        self.params = np.linalg.pinv(self.ata) @ self.atb

    def print_equation(self, normaliser):
        denormed = normaliser.denormalise_weights(self.params)
        p = [v.item() for v in denormed.flatten()]
        print(f"{p[0]:.2f}*{self.collumn_headers[0]}", end="")
        for i in range(1, self.num_params - 1):
            print(f" + {p[i]:.2f}*{self.collumn_headers[i]}", end="")
        print(f" + {p[-1]:.2f}")

# ---------------------------------------------------------
# HARDWARE ENGINE (Requires D=13)
# ---------------------------------------------------------

class HardwareLR:
    D = 13  # 12 features + bias
    FIELD_WIDTH = 18
    FIELD_MASK = (1 << FIELD_WIDTH) - 1
    NUM_WORDS = 16

    # ap_fixed<18,13> max ≈ 4096; worst-case ATA diagonal per sample ≈ 255
    # safe batch: floor(4096 / 255) = 16
    HW_BATCH_SIZE = 16

    ADDR_AP_CTRL     = 0x000
    ADDR_MEM_IN_DATA = 0x010
    ADDR_NUM_SAMPLES = 0x01c
    ADDR_ATB_BASE    = 0x040
    ADDR_ATA_BASE    = 0x400

    def __init__(self, ip, column_headers, bias_scale=1, max_samples=32768):
        self.ip = ip
        self.column_headers = column_headers
        self.num_params = len(self.column_headers)
        self.bias_scale = bias_scale
        self.weights = np.zeros((self.D, 1))
        self.max_samples = max_samples
        self._mem_in = allocate(shape=(max_samples, self.NUM_WORDS), dtype=np.int32)
        self._mem_addr_lo = int(self._mem_in.device_address) & 0xFFFFFFFF
        self._mem_addr_hi = (int(self._mem_in.device_address) >> 32) & 0xFFFFFFFF
        self.ata = np.zeros((self.num_params, self.num_params))
        self.atb = np.zeros((self.num_params, 1))

    def _run_hw_batch(self, test_x_int, test_y_int):
        """Send a small batch to the FPGA and accumulate ATA/ATB in float64."""
        n = len(test_y_int)
        self._mem_in[:n, 0] = test_y_int
        self._mem_in[:n, 1:14] = test_x_int
        self._mem_in.flush()

        self.ip.write(self.ADDR_NUM_SAMPLES, n)
        self.ip.write(self.ADDR_MEM_IN_DATA, self._mem_addr_lo)
        self.ip.write(self.ADDR_MEM_IN_DATA + 4, self._mem_addr_hi)
        self.ip.write(self.ADDR_AP_CTRL, 0x01)

        while not (self.ip.read(self.ADDR_AP_CTRL) & 0x02):
            pass

        hw_ata, hw_atb = self._read_hw_results()
        self.ata += hw_ata
        self.atb += hw_atb

    def _read_hw_results(self):
        ata_word_start = self.ADDR_ATA_BASE // 4
        ata_flat = np.array(
            self.ip.mmio.array[ata_word_start : ata_word_start + self.D ** 2],
            dtype=np.uint32
        )
        hw_ata = self._sign_extend_18(ata_flat).reshape(self.D, self.D)

        atb_word_start = self.ADDR_ATB_BASE // 4
        atb_flat = np.array(
            self.ip.mmio.array[atb_word_start : atb_word_start + self.D],
            dtype=np.uint32
        )
        hw_atb = self._sign_extend_18(atb_flat).reshape(self.D, 1)
        return hw_ata, hw_atb

    def _sign_extend_18(self, vals):
        masked = (vals & self.FIELD_MASK).astype(np.int64)
        masked[masked >= (1 << 17)] -= (1 << 18)
        return masked

    def compute_weights(self, ata, atb):
        return np.linalg.pinv(ata.astype(np.float64)) @ atb.astype(np.float64)

    def stream_chunk(self, samples_int):
        test_y = samples_int[:, -1].astype(np.int32)
        test_x = np.concatenate([
            samples_int[:, :-1],
            np.full((samples_int.shape[0], 1), self.bias_scale, dtype=np.int32)
        ], axis=1)

        n = len(test_y)
        for start in range(0, n, self.HW_BATCH_SIZE):
            end = min(start + self.HW_BATCH_SIZE, n)
            self._run_hw_batch(test_x[start:end], test_y[start:end])

        self.weights = self.compute_weights(self.ata, self.atb)

    def print_equation(self, normaliser):
        denormed = normaliser.denormalise_weights(self.weights, bias_scale=self.bias_scale)
        p = [v.item() for v in denormed.flatten()]
        print(f"{p[0]:.2f}*{self.column_headers[0]}", end="")
        for i in range(1, len(p) - 1):
            print(f" + {p[i]:.2f}*{self.column_headers[i]}", end="")
        print(f" + {p[-1]:.2f}")

# ---------------------------------------------------------
# MAIN ENGINES
# ---------------------------------------------------------

class LinearRegressionEngine:
    def __init__(self, ip):
        # 12 Features + 1 Bias = 13 Headers for the equations
        feat_names = list(FEATURE_RANGES.keys())[:-1] 
        collumn_headers = feat_names + ['BIAS']
        
        num_features = len(FEATURE_RANGES)
        self.normaliser = FeatureNormaliser(num_features, quant_bits=10)
        hw_bias_scale = int(round(self.normaliser.quant_max / 3.0))

        self.ip = ip
        self.unoptimised_sw_lr = UnoptimisedSoftwareLR(collumn_headers)
        self.optimised_sw_lr = OptimisedSoftwareLR(collumn_headers)
        self.hardware_lr = HardwareLR(ip, collumn_headers, bias_scale=hw_bias_scale)

    def preprocess_samples(self, samples):
        return self.normaliser.normalise_and_quantise(samples)

    def test_all_lr(self, ret_dict):
        samples_float = bundle_dict_to_numpy(ret_dict)
        samples_int = self.preprocess_samples(samples_float)

        t1 = time.time()
        self.unoptimised_sw_lr.stream_chunk(samples_int)
        t2 = time.time()
        self.optimised_sw_lr.stream_chunk_optimised(samples_int)
        t3 = time.time()
        self.hardware_lr.stream_chunk(samples_int)
        t4 = time.time()

        print(f"Samples: {len(samples_int)} | SW Unopt: {t2-t1:.4f}s | SW Opt: {t3-t2:.4f}s | HW: {t4-t3:.4f}s")

    def print_all_equations(self):
        print("\n[Optimised SW]")
        self.optimised_sw_lr.print_equation(self.normaliser)
        print("\n[UNOPTIMISED SW]")
        self.unoptimised_sw_lr.print_equation(self.normaliser)
        print("[HARDWARE FPGA]")
        self.hardware_lr.print_equation(self.normaliser)

class Engine:
    def __init__(self, ip):
        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()
        self.ce = CalculationEngine()
        self.ret_dict = defaultdict(list)
        self.lr_engine = LinearRegressionEngine(ip)
        
    def get_data(self):
        while True:
            now = time.time()
            # Safety: Snapshot deques before iterating
            trades_snapshot = list(self.binance_ws.trades) 
            shortterm_trades = self.binance_ws.get_trades_since(now - 1)
            
            vwma_10 = self.ce.vwma_calculate(trades_snapshot, now, 11.0)
            vwma_5 = self.ce.vwma_calculate(trades_snapshot, now, 6.0)
            total_sell = self.ce.sell_total(shortterm_trades, now, 1)
            total_brought = self.ce.bought_total(shortterm_trades, now, 1)
            stv = [t.volume for t in shortterm_trades]
            
            order_book = self.binance_ws.order_book
            asks = order_book.get("asks", [])
            bids = order_book.get("bids", [])

            if not asks or not bids:
                time.sleep(1)
                continue

            # Append exactly 13 items to keep dimensions aligned
            self.ret_dict["imballance"].append(0) 
            self.ret_dict["asks_total"].append(self.ce.price_depth(asks))
            self.ret_dict["bids_total"].append(self.ce.price_depth(bids))
            self.ret_dict["ask_dropoff"].append(self.ce.dropoff(asks, 5))
            self.ret_dict["bid_dropoff"].append(self.ce.dropoff(bids, 5))
            self.ret_dict["ask_spread"].append(self.ce.dropoff(asks, 3))
            self.ret_dict["vwma_10"].append(vwma_10)
            self.ret_dict["vwma_5"].append(vwma_5)
            self.ret_dict["total_sell"].append(total_sell)
            self.ret_dict["total_brought"].append(total_brought)
            self.ret_dict["stv"].append(sum(stv))
            self.ret_dict["last_price"].append(self.binance_ws.last_price)
            self.ret_dict["trade_arrival_rate"].append(len(shortterm_trades)) # Target
            
            time.sleep(1)

if __name__ == '__main__':
    # Assuming 'ip' is your PYNQ Overlay IP object
    eng = Engine(ip)
    threading.Thread(target=eng.get_data, daemon=True).start()

    try:
        while True:
            time.sleep(2)
            if len(eng.ret_dict["trade_arrival_rate"]) >= 15:
                # Thread-safe dictionary copy
                data_copy = {k: list(v) for k, v in eng.ret_dict.items()}
                eng.lr_engine.test_all_lr(data_copy)
                eng.lr_engine.print_all_equations()
                eng.ret_dict.clear()
            else:
                print(f"Warming up... {len(eng.ret_dict['trade_arrival_rate'])}/15")
    except KeyboardInterrupt:
        print("Shutting down...")
