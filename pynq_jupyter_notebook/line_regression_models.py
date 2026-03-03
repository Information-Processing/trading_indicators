
class UnoptimisedSoftwareLR:
    def __init__(self, collumn_headers):
        self.collumn_headers = collumn_headers
        self.num_params = len(self.collumn_headers)
        self.a = np.empty((0, self.num_params))
        self.b = np.empty((0, 1))
        self.params = None
    def solve(self):
        #solve function for doing (ATA)^-1 * (ATB)
        ata = self.a.T @ self.a
        atb = self.a.T @ self.b
        ata_inv = np.linalg.inv(ata)
        return ata_inv @ atb

    def print_equation(self):
        params = [p.item() for p in self.params]
        print(f"{params[0]:.2f} * {self.collumn_headers[0]}", end="")
        for i in range(1, self.num_params):
            print(f" + {params[i]:.2f} * {self.collumn_headers[i]}", end="")
        print("\n")

    def stream_chunk(self, lines):
        #stream chunk resolves/retrains the linear regression equation every time we add samples to it
        new_a = np.concatenate([lines[:, :-1], np.ones((len(lines), 1))], axis=1)
        new_b = lines[:, -1].reshape(-1, 1)

        self.a = np.vstack([self.a, new_a])
        self.b = np.vstack([self.b, new_b])

        self.params = self.solve()

class OptimisedSoftwareLR:
    def __init__(self, collumn_headers): 
        #initial data is sxp matrix
        self.collumn_headers = collumn_headers

        self.num_params = len(collumn_headers) #actually equal to num_params + 1 as last collumn is output
        
        self.ata = np.zeros((self.num_params,self.num_params))
        self.atb = np.zeros((self.num_params,1))
        self.params = None
        self.ata_inv = None
        

    def print_equation(self):
        params = [param.item() for param in self.params]
        print(f"{params[0]:.2f} * {self.collumn_headers[0]}", end="")
        for param_idx in range(1, self.num_params):
            print(f" + {params[param_idx]:.2f} * {self.collumn_headers[param_idx]}", end="") 

        print("\n")

    def stream_line(self, line):
        line_output = line[-1]
        q_vec = np.append(line[:-1], 1).reshape(1, self.num_params)
        qt_vec = q_vec.reshape(self.num_params, 1)

        ata_diff = qt_vec @ q_vec
        atb_diff = line_output * qt_vec

        self.ata += ata_diff
        self.atb += atb_diff

    def stream_chunk(self, lines):
        for line in lines:
            self.stream_line(np.array(line))
        self.recalculate_params()

    def stream_chunk_optimised(self, lines):
        line_output = lines[:, -1]
        q_mat = np.concatenate([lines[:,:-1], np.ones((lines.shape[0], 1))], axis=1)
        qt_mat = q_mat.transpose()

        ata_diff = qt_mat @ q_mat
        atb_diff = qt_mat @ line_output.reshape(-1, 1)

        self.ata += ata_diff
        self.atb += atb_diff

        self.recalculate_params()

    def recalculate_params(self):
        self.ata_inv = np.linalg.inv(self.ata)
        self.params = self.ata_inv @ self.atb

class HardwareLR:

    D = 13                     # 12 features + bias
    FIELD_WIDTH = 18
    FIELD_MASK = (1 << FIELD_WIDTH) - 1
    NUM_WORDS = 16             # 512-bit bus (16 x 32-bit words)

    # Register Addresses
    ADDR_AP_CTRL     = 0x000
    ADDR_MEM_IN_DATA = 0x010
    ADDR_NUM_SAMPLES = 0x01c
    ADDR_ATB_BASE    = 0x040
    ADDR_ATA_BASE    = 0x400

    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------

    def __init__(self, ip, column_headers, max_samples=32768):

        self.ip = ip
        self.column_headers = column_headers
        self.weights = np.array([])
        self.max_samples = max_samples

        # DMA buffer (int32 because IP reads 32-bit aligned words)
        self._mem_in = allocate(
            shape=(max_samples, self.NUM_WORDS),
            dtype=np.int32,
            cacheable=True
        )

        self._mem_addr_lo = int(self._mem_in.device_address) & 0xFFFFFFFF
        self._mem_addr_hi = (int(self._mem_in.device_address) >> 32) & 0xFFFFFFFF

    # ---------------------------------------------------------
    # MAIN EXECUTION
    # ---------------------------------------------------------

    def run_hardware(self, test_x_int, test_y_int):

        n = len(test_y_int)
        if n > self.max_samples:
            raise ValueError(
                f"Sample count {n} exceeds pre-allocated buffer ({self.max_samples})"
            )

        # -----------------------------
        # PACKING (JUST COPY INTS)
        # -----------------------------
        start = time.time()

        self._mem_in[:n, 0] = test_y_int
        self._mem_in[:n, 1:14] = test_x_int

        self._mem_in.flush()

        end = time.time()
        print(f"packing time: {end-start}")

        # -----------------------------
        # EXECUTE IP
        # -----------------------------
        start = time.time()

        self.ip.write(self.ADDR_NUM_SAMPLES, n)
        self.ip.write(self.ADDR_MEM_IN_DATA, self._mem_addr_lo)
        self.ip.write(self.ADDR_MEM_IN_DATA + 4, self._mem_addr_hi)

        self.ip.write(self.ADDR_AP_CTRL, 0x01)

        # wait for done
        while not (self.ip.read(self.ADDR_AP_CTRL) & 0x02):
            pass

        end = time.time()
        print(f"compute time: {end-start}")

        hw_ata, hw_atb = self.read_hw_results()
        weights = self.compute_weights(hw_ata, hw_atb)

        return hw_ata, hw_atb, weights

    # ---------------------------------------------------------
    # READ RESULTS FROM IP
    # ---------------------------------------------------------

    def read_hw_results(self):

        # Read ATA (D x D)
        ata_word_start = self.ADDR_ATA_BASE // 4

        ata_flat = np.array(
            self.ip.mmio.array[
                ata_word_start : ata_word_start + self.D * self.D
            ],
            dtype=np.uint32
        )

        hw_ata = self._sign_extend_18(ata_flat).reshape(self.D, self.D)

        # Read ATB (D)
        atb_word_start = self.ADDR_ATB_BASE // 4

        atb_flat = np.array(
            self.ip.mmio.array[
                atb_word_start : atb_word_start + self.D
            ],
            dtype=np.uint32
        )

        hw_atb = self._sign_extend_18(atb_flat)

        return hw_ata, hw_atb

    # ---------------------------------------------------------
    # SIGN EXTEND 18-BIT TWO'S COMPLEMENT
    # ---------------------------------------------------------

    def _sign_extend_18(self, vals):

        masked = (vals & self.FIELD_MASK).astype(np.int64)

        masked[masked >= (1 << 17)] -= (1 << 18)

        return masked

    # ---------------------------------------------------------
    # SOLVE REGRESSION
    # ---------------------------------------------------------

    def compute_weights(self, ata, atb):

        # Hardware already does >>5 shift internally
        # So we just solve normally
        return np.linalg.inv(
            ata.astype(np.float64)
        ) @ atb.astype(np.float64)

    # ---------------------------------------------------------
    # STREAM ENTRY POINT (CALLED BY ENGINE)
    # ---------------------------------------------------------

    def stream_chunk(self, samples_int):

        test_y = samples_int[:, -1].astype(np.int32)

        test_x = np.concatenate(
            [
                samples_int[:, :-1],
                np.ones((samples_int.shape[0], 1), dtype=np.int32)
            ],
            axis=1
        )

        _, _, weights = self.run_hardware(test_x, test_y)
        self.weights = weights

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def cleanup(self):
        if self._mem_in is not None:
            self._mem_in.freebuffer()
            self._mem_in = None

    # ---------------------------------------------------------
    # PRINT EQUATION
    # ---------------------------------------------------------

    def print_equation(self):

        weights = [param.item() for param in self.weights]

        print(f"{weights[0]:.2f} * {self.column_headers[0]}", end="")

        for weight_idx in range(1, len(self.weights)):
            print(
                f" + {weights[weight_idx]:.2f} * {self.column_headers[weight_idx]}",
                end=""
            )

        print("\n")

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
    "trade_arrival_rate" : 0,
}

def bundle_dict_to_numpy(data_dict):
    """
    Converts the entire ret_dict into a 2D numpy array 
    where rows are samples and columns are features.
    """
    # 1. Get all lists from the dictionary values
    cols = list(data_dict.values())
    
    # 2. Stack them side-by-side
    # This creates an array of shape (n_samples, n_features)
    feature_matrix = np.column_stack(cols)
    
    return feature_matrix


class LinearRegressionEngine:
    def __init__(self, ip):

        collumn_headers = list(FEATURE_RANGES.keys()) + ['1']

        self.ip = ip
        self.scale = 32


        self.unoptimised_sw_lr = self.initialise_unoptimised_sw_lr(collumn_headers)
        self.optimised_sw_lr = self.initialise_optimised_sw_lr(collumn_headers)
        self.hardware_lr = self.initialise_hardware_lr(ip, collumn_headers)

    # ---------------------------------------------------------
    # INITIALISERS
    # ---------------------------------------------------------

    def initialise_optimised_sw_lr(self, collumn_headers):
        return OptimisedSoftwareLR(collumn_headers)

    def initialise_unoptimised_sw_lr(self, collumn_headers):
        return UnoptimisedSoftwareLR(collumn_headers)

    def initialise_hardware_lr(self, ip, collumn_headers):
        return HardwareLR(ip, collumn_headers)

    # ---------------------------------------------------------
    # PREPROCESS ONCE HERE
    # ---------------------------------------------------------

    def preprocess_samples(self, samples):
        """
        Quantise float samples once.
        Return int32 array.
        """
        return np.rint(samples * self.scale).astype(np.int32)

    # ---------------------------------------------------------
    # TEST FUNCTIONS
    # ---------------------------------------------------------

    def test_unoptimised_sw_lr(self, samples):
        self.unoptimised_sw_lr.stream_chunk(samples)

    def test_optimised_sw_lr(self, samples):
        self.optimised_sw_lr.stream_chunk_optimised(samples)
    
    def test_hardware_lr(self, samples):
        self.hardware_lr.stream_chunk(samples)

    # ---------------------------------------------------------
    # MAIN BENCHMARK
    # ---------------------------------------------------------

    def test_all_lr(self, ret_dict):

        num_samples = 1
        
        print(f"dict size:{len(ret_dict.keys())}")
        print(len(ret_dict["trade_arrival_rate"]))
    
        if len(ret_dict["trade_arrival_rate"]) < 10:
            return
        samples_float = bundle_dict_to_numpy(ret_dict)

        samples_int = self.preprocess_samples(samples_float)

        print(f"TESTING {num_samples} samples:\n\n")

        # -----------------------------
        # Unoptimised software
        # -----------------------------
        t1 = time.time()
        self.test_unoptimised_sw_lr(samples_int)
        t2 = time.time()

        # -----------------------------
        # Optimised software
        # -----------------------------
        self.test_optimised_sw_lr(samples_int)
        t3 = time.time()

        # -----------------------------
        # Hardware
        # -----------------------------
        self.test_hardware_lr(samples_int)
        t4 = time.time()

        print("time for unoptimised software:") 
        print(f"{t2-t1}\n")

        print("time for optimised software:") 
        print(f"{t3-t2}\n")

        print("time for optimised hardware:") 
        print(f"{t4-t3}\n")

        times = (t4-t3)/min(t2-t1, t3-t2)
        print(f"slower by: {times}")

        return times

    # ---------------------------------------------------------

    def print_all_equations(self):

        print(f"\nunoptimised software equation:")
        self.unoptimised_sw_lr.print_equation()

        print(f"\noptimised software equation:")
        self.optimised_sw_lr.print_equation()

        print(f"\noptimised hardware equation:")
        self.hardware_lr.print_equation()
"""
if __name__ == "__main__":
    lr_engine = LinearRegressionEngine(ip)
    lr_engine.test_all_lr(10100)
    lr_engine.print_all_equations()
    
    min_time = float('inf')
    for i in range(100, 14000, 1000):
        calc_time = lr_engine.test_all_lr(i)
        
        if (i % 1000 == 0):
            print(i)
        if calc_time < min_time:
            min_time = calc_time
            min_idx = i
    print(min_time, min_idx)
    #lr_engine.print_all_equations()
   
"""
