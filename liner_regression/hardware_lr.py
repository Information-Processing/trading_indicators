import time
import numpy as np

class HardwareLR:
     
    # num of input parameters
    D = 13
    
    # num of input parameters + size for target
    NUM_FIELDS = D + 1
    
    QUANTIZE_SCALE = 32
    
    # bit width of fixed_t in outer_product.cpp. maximum bit width that fits in 1 DSP
    FIELD_WIDTH = 18
    
    FIELD_MASK = (1 << FIELD_WIDTH) - 1
    
    WORD_WIDTH = 32 #word width in input stream
    NUM_WORDS = 16 #number of words that can fit in 512 bit wide input stream

    # reg addresses as described in vitis drivers
    ADDR_AP_CTRL     = 0x000
    ADDR_MEM_IN_DATA = 0x010
    ADDR_NUM_SAMPLES = 0x01c
    ADDR_ATB_BASE    = 0x040
    ADDR_ATA_BASE    = 0x400

    def __init__(self, ip, collumn_headers):
        """ init hardware linear regression class and compute bit packing layout"""
        self.ip = ip

        self.collumn_headers = collumn_headers
        self.weights = np.array([])
        
        #list of where each word in the input stream starts
        self._bit_starts = np.arange(self.NUM_FIELDS) * self.FIELD_WIDTH

        self._word_lo    = self._bit_starts // self.WORD_WIDTH
        self._shift_lo   = self._bit_starts % self.WORD_WIDTH
        self._wraps      = (self._shift_lo + self.FIELD_WIDTH) > self.WORD_WIDTH
        self._word_hi    = self._word_lo + 1


    def pack_samples(self, test_x, test_y):
        """
        Vectorised packing of float features into 16 x uint32 hardware format.

        Layout per sample: [y(18b) | x0(18b) | x1(18b) | ... | x12(18b)]
        packed LSB-first into 16 x 32-bit words.
        """
        n = len(test_y)

        quantized = np.zeros((n, self.NUM_FIELDS), dtype=np.uint64)
        quantized[:, 0]  = np.round(test_y * self.QUANTIZE_SCALE).astype(np.int64) & self.FIELD_MASK
        quantized[:, 1:] = np.round(test_x * self.QUANTIZE_SCALE).astype(np.int64) & self.FIELD_MASK

        # scatter each 18-bit field into the correct 32-bit word positions
        mem = np.zeros((n, self.NUM_WORDS), dtype=np.uint64)
        for k in range(self.NUM_FIELDS):
            mem[:, self._word_lo[k]] |= (quantized[:, k] << int(self._shift_lo[k])) & 0xFFFFFFFF
            if self._wraps[k]:
                mem[:, self._word_hi[k]] |= quantized[:, k] >> int(self.WORD_WIDTH - self._shift_lo[k])

        return mem.astype(np.uint32)


    def read_hw_results(self, ip):
        """Read ATA (DxD) and ATB (D) from hardware registers."""
        
        #output ATA values interpreted as unsigned integers to a list (note that we are reading D*D times)
        ata_flat = np.array(
            [ip.read(self.ADDR_ATA_BASE + i * 4) for i in range(self.D * self.D)], dtype=np.uint32
        )

        #reshape the list as a (DXD) matrix
        hw_ata = ata_flat.view(np.int32).reshape(self.D, self.D)

        #output ATB values interpreted as unsigned integers to a list
        atb_flat = np.array(
            [ip.read(self.ADDR_ATB_BASE + i * 4) for i in range(self.D)], dtype=np.uint32
        )

        hw_atb = atb_flat.view(np.int32)

        return hw_ata, hw_atb


    def compute_weights(self, ata, atb):
        """
        Solve for regression weights using the following equation:
        X = inv(ATA) @ ATB

        Note that the quantisation scaling cancels as inv(s*ATA) @ (s*A'B) = inv(ATA) @ ATB
        """
        #convert ATA and ATB to floats
        ata_f = ata.astype(np.float64)
        atb_f = atb.astype(np.float64)
        
        ata_inv = np.linalg.inv(ata_f)

        x = ata_inv @ atb_f

        return x


    def run_hardware(self, ip, test_x, test_y):
        """Full pipeline: pack, execute on FPGA, read results, compute weights."""
        #number of samples
        n = len(test_y)
        
        #allocate memory buffer for all samples
        mem_in = allocate(shape=(n, self.NUM_WORDS), dtype=np.uint32)
        mem_in[:] = self.pack_samples(test_x, test_y)

        #write number of samples for FPGA
        ip.write(self.ADDR_NUM_SAMPLES, n)
        
        #stream samples into FPGA
        ip.write(self.ADDR_MEM_IN_DATA, mem_in.device_address & 0xFFFFFFFF)
        ip.write(self.ADDR_MEM_IN_DATA + 4, (mem_in.device_address >> 32) & 0xFFFFFFFF)
        
        #start mult operation
        ip.write(self.ADDR_AP_CTRL, 0x01)

        #spin lock to wait for compute
        while not (ip.read(self.ADDR_AP_CTRL) & 0x02):
            pass

        #read and process
        hw_ata, hw_atb = self.read_hw_results(ip)
        
        weights = self.compute_weights(hw_ata, hw_atb)
        
        #termination stage
        mem_in.freebuffer()
        return hw_ata, hw_atb, weights

    def test_hardware(self, test_x, test_y):
        """Software-only ATA/ATB (quantised, >> 5 to match hardware)."""
        xq = np.round(test_x * self.QUANTIZE_SCALE).astype(np.int64)
        yq = np.round(test_y * self.QUANTIZE_SCALE).astype(np.int64)

        sw_ata = (xq.T @ xq) >> 5
        sw_atb = (xq.T @ yq) >> 5
        return sw_ata, sw_atb


    def stream_chunk(self, ip, samples):
        """Run hardware + software, compare, and print weights."""
        
        test_y = samples[:, -1]
        test_x = np.concatenate([samples[:,:-1], np.ones((samples.shape[0], 1))], axis=1)

        hw_ata, hw_atb, weights = self.run_hardware(ip, test_x, test_y)

        self.weights = weights

    def print_equation(self):
        weights = [param.item() for param in self.weights]
        print(f"{weights[0]:.2f} * {self.collumn_headers[0]}", end="")
        for weight_idx in range(1, len(self.weights)):
            print(f" + {weights[weight_idx]:.2f} * {self.collumn_headers[weight_idx]}", end="") 

        print("\n")
