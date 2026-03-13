"""
Linear Regression engine using source2.cpp hardware.

All three methods (SW-unoptimised, SW-optimised, HW) receive the same
z-score normalised data (clipped to [-3, 3]). This ensures:
  1. Values fit inside ap_fixed<18,13> without saturation.
  2. All methods accumulate AtA/Atb in the same space, staying consistent.

source2.cpp converts the normalised float32 values to ap_fixed<18,13> in
hardware — no integer quantisation in Python. Accumulators are acc_t
(ap_fixed<64,40>). Weights are denormalised back to original units for output.
"""

from binance_ws import BinanceWSClient
from calc_engine_v2 import CalculationEngineV2
from pynq import allocate
import time
from collections import defaultdict
import numpy as np
import threading
from collections import deque
import requests

API_BASE_URL = "http://13.60.162.169:5000"

FEATURE_RANGES = {
    "spread_bps":           0,
    "wmid_deviation":       0,
    "book_imbalance":       0,
    "book_slope_ratio":     0,
    "depth_ratio":          0,
    "vol_delta_ratio":      0,
    "trade_intensity_z":    0,
    "large_trade_ratio":    0,
    "realized_volatility":  0,
    "momentum_10s":         0,
    "vwma_deviation_30s":   0,
    "cum_volume_delta":     0,
    "last_price":           0
}

def bundle_dict_to_numpy(data_dict):
    """Convert ret_dict into a 2D float64 array ordered by FEATURE_RANGES."""
    keys = list(FEATURE_RANGES.keys())
    min_len = min(len(data_dict[k]) for k in keys)
    cols = [data_dict[k][:min_len] for k in keys]
    return np.column_stack(cols).astype(np.float64)


# ---------------------------------------------------------
# FLOAT NORMALISER (z-score only — no integer quantisation)
# ---------------------------------------------------------

class FloatNormaliser:
    """
    Streaming z-score normaliser with Welford's online algorithm.
    Normalises to [-3, 3] range so values fit inside ap_fixed<18,13>.
    Unlike FeatureNormaliser, this does NOT quantise to integers —
    source2.cpp handles the float-to-fixed conversion in hardware.
    """

    def __init__(self, num_features):
        self.num_features = num_features
        self.n = 0
        self._mean = np.zeros(num_features)
        self._m2 = np.zeros(num_features)

    def _update_stats(self, batch):
        batch_n = batch.shape[0]
        if batch_n == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_m2 = batch.var(axis=0, ddof=0) * batch_n
        if self.n == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
        else:
            total_n = self.n + batch_n
            delta = batch_mean - self._mean
            self._mean += delta * (batch_n / total_n)
            self._m2 += batch_m2 + delta ** 2 * (self.n * batch_n / total_n)
        self.n += batch_n

    @property
    def std(self):
        if self.n < 2:
            return np.ones(self.num_features)
        s = np.sqrt(self._m2 / (self.n - 1))
        s[s < 1e-10] = 1.0
        return s

    def normalise(self, samples):
        """Z-score normalise and clip to [-3, 3]. Returns float64."""
        self._update_stats(samples)
        z = (samples - self._mean) / self.std
        return np.clip(z, -3.0, 3.0)

    def denormalise_weights(self, weights_norm):
        """
        Convert weights from normalised space back to original units.

        In normalised space:  ỹ = Σ(w̃_i · x̃_i) + w̃_bias
        where x̃_i = (x_i − μ_i) / σ_i,  ỹ = (y − μ_y) / σ_y

        In original space:   y = Σ(w_i · x_i) + b
        where w_i = w̃_i · σ_y / σ_i
              b   = w̃_bias · σ_y + μ_y − Σ(w_i · μ_i)
        """
        orig_shape = weights_norm.shape
        w = weights_norm.flatten()

        feat_std = self.std[:-1]
        target_std = self.std[-1]
        feat_mean = self._mean[:-1]
        target_mean = self._mean[-1]

        w_feat = w[:-1] * (target_std / feat_std)
        w_bias = w[-1] * target_std + target_mean - np.sum(w_feat * feat_mean)

        return np.concatenate([w_feat, [w_bias]]).reshape(orig_shape)


# ---------------------------------------------------------
# SOFTWARE LINEAR REGRESSION (normalised floats)
# ---------------------------------------------------------

class UnoptimisedSoftwareLR:
    """Naive LR that stores full A and b matrices."""

    def __init__(self, column_headers):
        self.column_headers = column_headers
        self.num_params = len(column_headers)
        self.a = np.empty((0, self.num_params))
        self.b = np.empty((0, 1))
        self.params = np.zeros((self.num_params, 1))

    def solve(self):
        ata = self.a.T @ self.a
        atb = self.a.T @ self.b
        return np.linalg.pinv(ata) @ atb

    def stream_chunk(self, lines):
        new_a = np.concatenate([lines[:, :-1], np.ones((len(lines), 1))], axis=1)
        new_b = lines[:, -1].reshape(-1, 1)
        self.a = np.vstack([self.a, new_a])
        self.b = np.vstack([self.b, new_b])
        self.params = self.solve()


class OptimisedSoftwareLR:
    """Streaming LR that only maintains AtA and Atb accumulators."""

    def __init__(self, column_headers):
        self.column_headers = column_headers
        self.num_params = len(column_headers)
        self.ata = np.zeros((self.num_params, self.num_params))
        self.atb = np.zeros((self.num_params, 1))
        self.params = np.zeros((self.num_params, 1))

    def stream_chunk_optimised(self, lines):
        line_output = lines[:, -1]
        q_mat = np.concatenate([lines[:, :-1], np.ones((lines.shape[0], 1))], axis=1)
        qt_mat = q_mat.T
        self.ata += qt_mat @ q_mat
        self.atb += qt_mat @ line_output.reshape(-1, 1)
        self.recalculate_params()

    def recalculate_params(self):
        self.params = np.linalg.pinv(self.ata) @ self.atb


# ---------------------------------------------------------
# HARDWARE LINEAR REGRESSION (source2.cpp — float input)
# ---------------------------------------------------------

class HardwareLR:
    """
    Drives source2.cpp on the FPGA.

    Data is sent as raw IEEE float32 bit-patterns inside a 512-bit wide bus.
    The hardware casts each float to ap_fixed<18,13> internally and accumulates
    AtA / Atb in acc_t (ap_fixed<64,40>).
    """

    D = 13          # 12 features + 1 bias
    NUM_WORDS = 16  # 512-bit bus = 16 × 32-bit words

    # ap_fixed<64,40>: 40 integer bits, 24 fractional bits
    ACC_FRAC_BITS = 24

    # acc_t max ≈ 2^39;  worst-case product (4096^2) ≈ 16.7 M
    # Safe samples per kernel call ≈ 32 000
    HW_BATCH_SIZE = 4096

    # ── AXI-Lite register map (source2.cpp with acc_t outputs) ──
    # Vitis HLS aligns arrays to the next power-of-2 of their byte size:
    #   Atb: 13×8 = 104 bytes  → aligned to 128  → base 0x080
    #   AtA: 169×8 = 1352 bytes → aligned to 2048 → base 0x800
    ADDR_AP_CTRL     = 0x000
    ADDR_MEM_IN_DATA = 0x010
    ADDR_NUM_SAMPLES = 0x01c
    ADDR_ATB_BASE    = 0x080   # D   × acc_t (64-bit, 2 words each)
    ADDR_ATA_BASE    = 0x800   # D*D × acc_t (64-bit, 2 words each)

    def __init__(self, ip, column_headers,
                 max_samples=32768, addr_atb=None, addr_ata=None):
        self.ip = ip
        self.column_headers = column_headers
        self.num_params = len(column_headers)
        self.weights = np.zeros((self.D, 1))
        self.max_samples = max_samples

        if addr_atb is not None:
            self.ADDR_ATB_BASE = addr_atb
        if addr_ata is not None:
            self.ADDR_ATA_BASE = addr_ata

        self._mem_in = allocate(shape=(max_samples, self.NUM_WORDS), dtype=np.int32)
        self._mem_addr_lo = int(self._mem_in.device_address) & 0xFFFFFFFF
        self._mem_addr_hi = (int(self._mem_in.device_address) >> 32) & 0xFFFFFFFF

        self.ata = np.zeros((self.num_params, self.num_params), dtype=np.float64)
        self.atb = np.zeros((self.num_params, 1), dtype=np.float64)

    # ── packing helpers ──

    @staticmethod
    def _floats_to_bits(arr):
        """Reinterpret float32 array as int32 (preserves IEEE 754 bit-pattern)."""
        return np.ascontiguousarray(arr, dtype=np.float32).view(np.int32)

    # ── kernel invocation ──

    def _run_hw_batch(self, x_float, y_float):
        """Pack one batch into DMA buffer, launch kernel, accumulate results."""
        n = len(y_float)

        self._mem_in[:n, 0] = self._floats_to_bits(y_float)
        self._mem_in[:n, 1:self.D + 1] = self._floats_to_bits(x_float)
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

    # ── reading 64-bit acc_t results ──

    def _read_64bit_array(self, base_addr, count):
        """
        Read *count* acc_t values (ap_fixed<64,40>) from consecutive
        AXI-Lite registers. Each value spans two 32-bit words (lo, hi).
        """
        word_start = base_addr // 4
        raw = np.array(
            self.ip.mmio.array[word_start : word_start + count * 2],
            dtype=np.uint32,
        )
        lo = raw[0::2].astype(np.int64)
        hi = raw[1::2].astype(np.int64)
        combined = (hi << 32) | lo   # two's-complement reconstruction
        return combined.astype(np.float64) / (1 << self.ACC_FRAC_BITS)

    def _read_hw_results(self):
        hw_ata = self._read_64bit_array(
            self.ADDR_ATA_BASE, self.D * self.D
        ).reshape(self.D, self.D)

        hw_atb = self._read_64bit_array(
            self.ADDR_ATB_BASE, self.D
        ).reshape(self.D, 1)

        return hw_ata, hw_atb

    # ── public API ──

    def stream_chunk(self, samples_norm):
        """
        Accepts pre-normalised float samples with shape (N, 13).
        Last column is the target; first 12 are features.
        A bias column of 1.0 is appended automatically.
        """
        test_y = samples_norm[:, -1].astype(np.float32)
        test_x = np.concatenate([
            samples_norm[:, :-1].astype(np.float32),
            np.ones((samples_norm.shape[0], 1), dtype=np.float32),
        ], axis=1)

        n = len(test_y)
        for start in range(0, n, self.HW_BATCH_SIZE):
            end = min(start + self.HW_BATCH_SIZE, n)
            self._run_hw_batch(test_x[start:end], test_y[start:end])

        self.weights = np.linalg.pinv(self.ata) @ self.atb


# ---------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------

class LinearRegressionEngine:
    """
    Runs all three LR variants on the same normalised data.

    Raw floats are z-score normalised once, then fed identically to
    SW-unoptimised, SW-optimised, and HW. All three accumulate in the
    same normalised space, so their AtA/Atb stay consistent even as
    the normaliser statistics evolve. Weights are denormalised back
    to original units when printed.
    """

    def __init__(self, ip, enable_hardware):
        self.enable_hardware = enable_hardware
        feat_names = list(FEATURE_RANGES.keys())[:-1]
        self.column_headers = feat_names + ["BIAS"]
        self.normaliser = FloatNormaliser(len(FEATURE_RANGES))

        self.unoptimised_sw_lr = UnoptimisedSoftwareLR(self.column_headers)
        self.optimised_sw_lr = OptimisedSoftwareLR(self.column_headers)
        if self.enable_hardware:
            self.hardware_lr = HardwareLR(ip, self.column_headers, max_samples=32768)
        
    def _get_denormed(self, weights_norm):
        return self.normaliser.denormalise_weights(weights_norm).flatten().tolist()

    def test_all_lr(self, ret_dict):
        samples_float = bundle_dict_to_numpy(ret_dict)
        samples_norm = self.normaliser.normalise(samples_float)

        t1 = time.time()
        self.unoptimised_sw_lr.stream_chunk(samples_norm)
        t2 = time.time()
        self.optimised_sw_lr.stream_chunk_optimised(samples_norm)
        t3 = time.time()
        if self.enable_hardware: 
            self.hardware_lr.stream_chunk(samples_norm)
        t4 = time.time()

        print(
            f"Samples: {len(samples_norm)} | "
            f"SW Unopt: {t2-t1:.4f}s | SW Opt: {t3-t2:.4f}s | HW: {t4-t3:.4f}s"
        )

    def _print_denormed(self, weights_norm):
        denormed = self.normaliser.denormalise_weights(weights_norm)
        p = denormed.flatten()
        parts = [f"{p[0]:.6g}*{self.column_headers[0]}"]
        for i in range(1, len(p) - 1):
            parts.append(f"{p[i]:.6g}*{self.column_headers[i]}")
        print(" + ".join(parts) + f" + {p[-1]:.6g}")

    def print_all_equations(self):
        print("\n[Optimised SW]")
        self._print_denormed(self.optimised_sw_lr.params)
        print("\n[UNOPTIMISED SW]")
        self._print_denormed(self.unoptimised_sw_lr.params)
        print("\n[HARDWARE FPGA]")
        if self.enable_hardware:
            self._print_denormed(self.hardware_lr.weights)
        
    def post_weights(self, asset: str):
        """POST denormalised weights + feature names to the API server."""
        weights = self._get_denormed(self.optimised_sw_lr.params)
        features = self.column_headers  # 12 features + "BIAS"
        payload = {
            "weights": weights,
            "features": features,
            "asset": asset,
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/matrix", json=payload, timeout=5)
            resp.raise_for_status()
            print(f"[API] POST /matrix OK — {resp.status_code}")
        except requests.RequestException as e:
            print(f"[API] POST /matrix FAILED — {e}")

    def get_weights(self):
        return self._get_denormed(self.optimised_sw_lr.params)

POLLING_PERIOD = 0.2
WARMUP_PERIOD = 5
WARMUP_ITERATIONS = WARMUP_PERIOD / POLLING_PERIOD 
class Engine:
    def __init__(self, ip):
        self.enable_hardware = False 

        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()
        self.ce = CalculationEngineV2()
        self.ret_dict = defaultdict(list)
        self.lr_engine = LinearRegressionEngine(ip, self.enable_hardware)
        self.last_price_queue = deque()

    def get_data(self):
        iterations = 0
        trades = self.binance_ws.trades
        last_price = self.binance_ws.last_price

        while(1):
            now = time.time()
            trades_snapshot = list(self.binance_ws.trades)
            trades_10 = self.binance_ws.get_trades_since(now - 11.0)
            trades_30 = self.binance_ws.get_trades_since(now - 31.0)
            shortterm_trades = self.binance_ws.get_trades_since(now - 1)

            order_book = self.binance_ws.order_book
            asks = order_book.get("asks", [])
            bids = order_book.get("bids", [])

            if not asks or not bids:
                time.sleep(1)
                continue

            best_ask = asks[0]   # (price, qty)
            best_bid = bids[0]   # (price, qty)
            last_price = self.binance_ws.last_price

            # --- 12 new features ---
            spread      = self.ce.spread_bps(best_ask[0], best_bid[0])
            wmid_dev    = self.ce.weighted_mid_deviation(best_ask, best_bid)
            imbalance   = self.ce.book_imbalance(asks, bids, levels=10)
            slope_ratio = self.ce.book_slope_ratio(asks, bids, levels=5)
            depth_r     = self.ce.depth_ratio(asks, bids)
            vol_delta   = self.ce.volume_delta_ratio(trades_snapshot, now, 5.0)
            intensity   = self.ce.trade_intensity_zscore(len(shortterm_trades))
            large_ratio = self.ce.large_trade_ratio(trades_snapshot, now, 5.0, threshold_qty=0.1)
            volatility  = self.ce.realized_volatility(trades_10, now, 10.0)
            mom         = self.ce.momentum(trades_10, now, 10.0)
            vwma_dev    = self.ce.vwma_deviation(trades_30, now, 30.0, last_price)
            cum_vdelta  = self.ce.cumulative_volume_delta(vol_delta)

            self.ret_dict["spread_bps"].append(spread)
            self.ret_dict["wmid_deviation"].append(wmid_dev)
            self.ret_dict["book_imbalance"].append(imbalance)
            self.ret_dict["book_slope_ratio"].append(slope_ratio)
            self.ret_dict["depth_ratio"].append(depth_r)
            self.ret_dict["vol_delta_ratio"].append(vol_delta)
            self.ret_dict["trade_intensity_z"].append(intensity)
            self.ret_dict["large_trade_ratio"].append(large_ratio)
            self.ret_dict["realized_volatility"].append(volatility)
            self.ret_dict["momentum_10s"].append(mom)
            self.ret_dict["vwma_deviation_30s"].append(vwma_dev)
            self.ret_dict["cum_volume_delta"].append(cum_vdelta)
            

            if iterations > WARMUP_ITERATIONS:
                self.last_price_queue.append(self.binance_ws.last_price)
                self.ret_dict["last_price"].append(self.last_price_queue.popleft())
            iterations += 1

            time.sleep(POLLING_PERIOD)



class TestingEngine:
    def __init__(self):
        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()
        self.ce = CalculationEngineV2()
        self.last_last_price = 0
        self.last_prediction = 0

    def use_weights(self, weights):
        trades = self.binance_ws.trades
        last_price = self.binance_ws.last_price

        now = time.time()
        trades_snapshot = list(self.binance_ws.trades)
        trades_10 = self.binance_ws.get_trades_since(now - 11.0)
        trades_30 = self.binance_ws.get_trades_since(now - 31.0)
        shortterm_trades = self.binance_ws.get_trades_since(now - 1)

        order_book = self.binance_ws.order_book
        asks = order_book.get("asks", [])
        bids = order_book.get("bids", [])

        best_ask = asks[0]   # (price, qty)
        best_bid = bids[0]   # (price, qty)
        last_price = self.binance_ws.last_price

        indicators = {}
        # --- 12 new features ---
        indicators["spread"] = self.ce.spread_bps(best_ask[0], best_bid[0])
        indicators["wmid_dev"] = self.ce.weighted_mid_deviation(best_ask, best_bid)
        indicators["imbalance"] = self.ce.book_imbalance(asks, bids, levels=10)
        indicators["slope_ratio"] = self.ce.book_slope_ratio(asks, bids, levels=5)
        indicators["depth_r"] = self.ce.depth_ratio(asks, bids)
        indicators["vol_delta"] = self.ce.volume_delta_ratio(trades_snapshot, now, 5.0)
        indicators["intensity"] = self.ce.trade_intensity_zscore(len(shortterm_trades))
        indicators["large_ratio"] = self.ce.large_trade_ratio(trades_snapshot, now, 5.0, threshold_qty=0.1)
        indicators["volatility"] = self.ce.realized_volatility(trades_10, now, 10.0)
        indicators["mom"] = self.ce.momentum(trades_10, now, 10.0)
        indicators["vwma_dev"] = self.ce.vwma_deviation(trades_30, now, 30.0, last_price)
        indicators["cum_vdelta"] = self.ce.cumulative_volume_delta(indicators["vol_delta"])
            
        
        print('='*100)
        pred = float(np.dot(weights[:-1], np.array(list(indicators.values()))) + weights[-1])
        print(f"prediction: {pred}, actual price: {self.binance_ws.last_price}")
        prediction = ""
        if self.last_prediction - pred < 0:
            prediction = "SELL"
        else:
            prediction = "BUY"

        if self.last_last_price - self.binance_ws.last_price < 0:
            actual = "SELL"
        else:
            actual = "BUY"
        print(f"predicted signal: {prediction}, actual signal: {actual}")
        print('='*100)
        self.last_last_price = self.binance_ws.last_price
        self.last_prediction = pred

if __name__ == "__main__":
    ip = 0
    eng = Engine(ip)
    threading.Thread(target=eng.get_data, daemon=True).start()
    
    binance_testing = True
    test_binance = None
    if binance_testing:
        test_binance = TestingEngine()

    try:
        while True:
            time.sleep(2)
            if len(eng.ret_dict["cum_volume_delta"]) >= 50:
                target_samples = len(eng.ret_dict["last_price"])
                data_copy = {k : list(v[:target_samples]) for k, v in eng.ret_dict.items()}

                eng.lr_engine.test_all_lr(data_copy)
                eng.lr_engine.print_all_equations()
                eng.lr_engine.post_weights("BTC")  # ← upload to API
                eng.ret_dict.clear()

                if test_binance:
                    weights = eng.lr_engine.get_weights()
                    test_binance.use_weights(weights)

            else:
                print(f"Warming up... {len(eng.ret_dict['cum_volume_delta'])}/15")
    except KeyboardInterrupt:
        print("Shutting down...")
