import numpy as np
import time
from pynq import allocate


class HardwareLR:

    D = 13
    QUANTIZE_SCALE = 32
    FIELD_WIDTH = 18
    FIELD_MASK = (1 << FIELD_WIDTH) - 1
    NUM_WORDS = 16

    # reg addresses as described in vitis drivers
    ADDR_AP_CTRL     = 0x000
    ADDR_MEM_IN_DATA = 0x010
    ADDR_NUM_SAMPLES = 0x01c
    ADDR_ATB_BASE    = 0x040
    ADDR_ATA_BASE    = 0x400

    def __init__(self, ip, collumn_headers, max_samples=8192):
        """Init hardware linear regression with pre-allocated DMA buffer."""
        self.ip = ip
        self.collumn_headers = collumn_headers
        self.weights = np.array([])
        self.max_samples = max_samples

        self._mem_in = allocate(
            shape=(max_samples, self.NUM_WORDS), dtype=np.uint32, cacheable=True
        )
        self._mem_addr_lo = int(self._mem_in.device_address) & 0xFFFFFFFF
        self._mem_addr_hi = (int(self._mem_in.device_address) >> 32) & 0xFFFFFFFF

        # normalisation state
        self._x_mean = np.zeros(12)
        self._x_std  = np.ones(12)
        self._y_mean = 0.0
        self._y_std  = 1.0

    def run_hardware(self, ip, test_x, test_y):
        """Quantize, write to DMA buffer, execute on FPGA, read results.

        Assumes features and target are in [-1, 1].  Each value is written
        to its own 32-bit aligned slot in the 512-bit wide bus — no
        bit-packing required.
        """
        n = len(test_y)
        if n > self.max_samples:
            raise ValueError(
                f"Sample count {n} exceeds pre-allocated buffer ({self.max_samples})"
            )

        self._mem_in[:n, 0]    = np.round(test_y * self.QUANTIZE_SCALE).astype(np.int32).view(np.uint32)
        self._mem_in[:n, 1:14] = np.round(test_x * self.QUANTIZE_SCALE).astype(np.int32).view(np.uint32)
        self._mem_in.flush()

        start = time.time()
        ip.write(self.ADDR_NUM_SAMPLES, n)
        ip.write(self.ADDR_MEM_IN_DATA, self._mem_addr_lo)
        ip.write(self.ADDR_MEM_IN_DATA + 4, self._mem_addr_hi)

        ip.write(self.ADDR_AP_CTRL, 0x01)
        while not (ip.read(self.ADDR_AP_CTRL) & 0x02):
            pass
        end = time.time()
        print(f"compute time:{end-start}")

        hw_ata, hw_atb = self.read_hw_results(ip)
        weights = self.compute_weights(hw_ata, hw_atb)

        return hw_ata, hw_atb, weights

    def read_hw_results(self, ip):
        """Bulk-read ATA (DxD) and ATB (D) via MMIO array slice."""
        ata_word_start = self.ADDR_ATA_BASE // 4
        ata_flat = np.array(
            ip.mmio.array[ata_word_start : ata_word_start + self.D * self.D],
            dtype=np.uint32,
        )
        hw_ata = self._sign_extend_18(ata_flat).reshape(self.D, self.D)

        atb_word_start = self.ADDR_ATB_BASE // 4
        atb_flat = np.array(
            ip.mmio.array[atb_word_start : atb_word_start + self.D],
            dtype=np.uint32,
        )
        hw_atb = self._sign_extend_18(atb_flat)

        return hw_ata, hw_atb

    def _sign_extend_18(self, vals):
        """Sign-extend 18-bit two's complement values stored in uint32 registers."""
        masked = (vals & self.FIELD_MASK).astype(np.int64)
        masked[masked >= (1 << 17)] -= (1 << 18)
        return masked

    def compute_weights(self, ata, atb):
        """
        Solve for regression weights:  w = inv(ATA) @ ATB

        Quantisation scaling (x32) cancels:
            inv(32*X'X) @ (32*X'y) = inv(X'X) @ X'y

        Then denormalises from z-scored space back to original units.
        """
        w_norm = np.linalg.inv(ata.astype(np.float64)) @ atb.astype(np.float64)

        weights = np.empty(self.D, dtype=np.float64)
        weights[:12] = self._y_std * w_norm[:12] / self._x_std
        weights[12]  = (self._y_mean + self._y_std * (w_norm[12] - np.dot(w_norm[:12], self._x_mean / self._x_std)))
        return weights

    def test_hardware(self, test_x, test_y):
        """Software-only ATA/ATB (quantised, >> 5 to match hardware flush)."""
        xq = np.round(test_x * self.QUANTIZE_SCALE).astype(np.int64)
        yq = np.round(test_y * self.QUANTIZE_SCALE).astype(np.int64)

        sw_ata = (xq.T @ xq) >> 5
        sw_atb = (xq.T @ yq) >> 5
        return sw_ata, sw_atb

    def stream_chunk(self, ip, samples):
        """Run hardware and compute weights for a batch of samples."""
        test_y = samples[:, -1]
        test_x = np.concatenate(
            [samples[:, :-1], np.ones((samples.shape[0], 1))], axis=1,
        )
        _, _, weights = self.run_hardware(ip, test_x, test_y)
        self.weights = weights

    def cleanup(self):
        """Free the pre-allocated DMA buffer."""
        if self._mem_in is not None:
            self._mem_in.freebuffer()
            self._mem_in = None

    def print_equation(self):
        weights = [param.item() for param in self.weights]
        print(f"{weights[0]:.2f} * {self.collumn_headers[0]}", end="")
        for weight_idx in range(1, len(self.weights)):
            print(f" + {weights[weight_idx]:.2f} * {self.collumn_headers[weight_idx]}", end="")
        print("\n")
