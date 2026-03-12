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
from calc_engine import CalculationEngine
from pynq import allocate
import time
from collections import defaultdict
import numpy as np
import threading

FEATURE_RANGES = {
    "imballance":         0,
    "asks_total":         0,
    "bids_total":         0,
    "ask_dropoff":        0,
    "bid_dropoff":        0,
    "ask_spread":         0,
    "vwma_10":            0,
    "vwma_5":             0,
    "total_sell":         0,
    "total_brought":      0,
    "stv":                0,
    "trade_arrival_rate": 0,  # 12th feature
    "last_price":         0,  # Target (Y)
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

    def __init__(self, ip):
        feat_names = list(FEATURE_RANGES.keys())[:-1]
        self.column_headers = feat_names + ["BIAS"]
        self.normaliser = FloatNormaliser(len(FEATURE_RANGES))

        self.unoptimised_sw_lr = UnoptimisedSoftwareLR(self.column_headers)
        self.optimised_sw_lr = OptimisedSoftwareLR(self.column_headers)
        self.hardware_lr = HardwareLR(ip, self.column_headers, max_samples=32768)

    def test_all_lr(self, ret_dict):
        samples_float = bundle_dict_to_numpy(ret_dict)
        samples_norm = self.normaliser.normalise(samples_float)

        t1 = time.time()
        self.unoptimised_sw_lr.stream_chunk(samples_norm)
        t2 = time.time()
        self.optimised_sw_lr.stream_chunk_optimised(samples_norm)
        t3 = time.time()
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
        self._print_denormed(self.hardware_lr.weights)


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
                time.sleep(0.3)
                continue

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
            self.ret_dict["trade_arrival_rate"].append(len(shortterm_trades))
            self.ret_dict["last_price"].append(self.binance_ws.last_price)

            time.sleep(0.2)


if __name__ == "__main__":
    eng = Engine(ip)
    threading.Thread(target=eng.get_data, daemon=True).start()

    try:
        while True:
            time.sleep(2)
            if len(eng.ret_dict["trade_arrival_rate"]) >= 15:
                data_copy = {k: list(v) for k, v in eng.ret_dict.items()}
                eng.lr_engine.test_all_lr(data_copy)
                eng.lr_engine.print_all_equations()
                eng.ret_dict.clear()
            else:
                print(f"Warming up... {len(eng.ret_dict['trade_arrival_rate'])}/15")
    except KeyboardInterrupt:
        print("Shutting down...")