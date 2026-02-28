import numpy as np
import time
from pynq import allocate

# --- Parameters ---
D = 13
NUM_FIELDS = D + 1
QUANTIZE_SCALE = 32
FIELD_WIDTH = 18
FIELD_MASK = (1 << FIELD_WIDTH) - 1
WORD_WIDTH = 32
NUM_WORDS = 16

# --- Register Addresses ---
ADDR_AP_CTRL     = 0x000
ADDR_MEM_IN_DATA = 0x010
ADDR_NUM_SAMPLES = 0x01c
ADDR_ATB_BASE    = 0x040
ADDR_ATA_BASE    = 0x400

# Precompute bit-packing layout (which 32-bit words each 18-bit field lands in)
_bit_starts = np.arange(NUM_FIELDS) * FIELD_WIDTH
_word_lo    = _bit_starts // WORD_WIDTH
_shift_lo   = _bit_starts % WORD_WIDTH
_wraps      = (_shift_lo + FIELD_WIDTH) > WORD_WIDTH
_word_hi    = _word_lo + 1


def pack_samples(test_x, test_y):
    """Vectorised packing of float features into 16 x uint32 hardware format.

    Layout per sample: [y(18b) | x0(18b) | x1(18b) | ... | x12(18b)]
    packed LSB-first into 16 x 32-bit words.
    """
    n = len(test_y)

    quantized = np.zeros((n, NUM_FIELDS), dtype=np.uint64)
    quantized[:, 0]  = np.round(test_y * QUANTIZE_SCALE).astype(np.int64) & FIELD_MASK
    quantized[:, 1:] = np.round(test_x * QUANTIZE_SCALE).astype(np.int64) & FIELD_MASK

    # scatter each 18-bit field into the correct 32-bit word positions
    mem = np.zeros((n, NUM_WORDS), dtype=np.uint64)
    for k in range(NUM_FIELDS):
        mem[:, _word_lo[k]] |= (quantized[:, k] << int(_shift_lo[k])) & 0xFFFFFFFF
        if _wraps[k]:
            mem[:, _word_hi[k]] |= quantized[:, k] >> int(WORD_WIDTH - _shift_lo[k])

    return mem.astype(np.uint32)


def read_hw_results(ip):
    """Read ATA (DxD) and ATB (D) from hardware registers."""
    ata_flat = np.array(
        [ip.read(ADDR_ATA_BASE + i * 4) for i in range(D * D)], dtype=np.uint32
    )
    hw_ata = ata_flat.view(np.int32).reshape(D, D)

    atb_flat = np.array(
        [ip.read(ADDR_ATB_BASE + i * 4) for i in range(D)], dtype=np.uint32
    )
    hw_atb = atb_flat.view(np.int32)

    return hw_ata, hw_atb


def compute_weights(ata, atb):
    """Solve for regression weights: w = inv(ATA) @ ATB.

    Quantisation scaling cancels: inv(s*X'X) @ (s*X'y) = inv(X'X) @ X'y
    """
    return np.linalg.inv(ata.astype(np.float64)) @ atb.astype(np.float64)


def run_hardware(ip, test_x, test_y):
    """Full pipeline: pack, execute on FPGA, read results, compute weights."""
    n = len(test_y)
    mem_in = allocate(shape=(n, NUM_WORDS), dtype=np.uint32)
    mem_in[:] = pack_samples(test_x, test_y)

    start = time.time()

    ip.write(ADDR_NUM_SAMPLES, n)
    ip.write(ADDR_MEM_IN_DATA, mem_in.device_address & 0xFFFFFFFF)
    ip.write(ADDR_MEM_IN_DATA + 4, (mem_in.device_address >> 32) & 0xFFFFFFFF)

    ip.write(ADDR_AP_CTRL, 0x01)
    while not (ip.read(ADDR_AP_CTRL) & 0x02):
        pass

    hw_ata, hw_atb = read_hw_results(ip)
    hw_time = time.time() - start
    print(f"Hardware time: {hw_time:.6f}s")

    weights = compute_weights(hw_ata, hw_atb)
    print(f"Weights:\n{weights}")

    mem_in.freebuffer()
    return hw_ata, hw_atb, weights


def software_reference(test_x, test_y):
    """Software-only ATA/ATB (quantised, >> 5 to match hardware)."""
    xq = np.round(test_x * QUANTIZE_SCALE).astype(np.int64)
    yq = np.round(test_y * QUANTIZE_SCALE).astype(np.int64)

    sw_ata = (xq.T @ xq) >> 5
    sw_atb = (xq.T @ yq) >> 5
    return sw_ata, sw_atb


def verify(ip, num_samples=7375):
    """Run hardware + software, compare, and print weights."""
    test_x = np.random.uniform(0.5, 1.5, (num_samples, D))
    test_y = np.random.uniform(0.5, 1.5, num_samples)

    hw_ata, hw_atb, weights = run_hardware(ip, test_x, test_y)

    start = time.time()
    sw_ata, sw_atb = software_reference(test_x, test_y)
    print(f"Software time: {time.time() - start:.6f}s")

    print(f"AtA sample (0,0) — HW: {hw_ata[0,0]}  SW: {sw_ata[0,0]}")
    print(f"Atb sample (0)   — HW: {hw_atb[0]}  SW: {sw_atb[0]}")
    print(f"AtA match: {np.allclose(hw_ata, sw_ata, atol=0)}")
    print(f"Atb match: {np.allclose(hw_atb, sw_atb, atol=0)}")

    return weights
