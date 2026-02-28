import numpy as np
import time

# example input:
# [[1,2,3,4,5],
#  [1,2,3,4,5]]

LOG_LVL = 1

def log(message, level):
    if (level <= LOG_LVL):
        print(message)


FEATURE_RANGES = {
    "VWAP":         (95.0, 105.0),
    "Vol":          (1.0, 10.0),
    "BookImb":      (0.3, 0.75),
    "Spread":       (0.05, 0.25),
    "BuySellRatio": (0.4, 2.5),
    "LogRet":       (-0.01, 0.01),
    "TradeRate":    (20.0, 70.0),
    "Volatility":   (0.03, 0.25),
    "CVD":          (-20.0, 45.0),
    "BidDepth":     (50.0, 500.0),
    "AskDepth":     (50.0, 500.0),
    "PriceRange":   (0.01, 2.0),
    "AvgTradeSize": (0.001, 0.5),
}

def generate_random_data(n_samples):
    cols = []
    for bounds in FEATURE_RANGES.values():
        cols.append(np.random.uniform(bounds[0], bounds[1], n_samples))

    features = np.column_stack(cols)
    #make the target (yval) a distribution of the inputs so its not completely rng
    true_weights = np.random.uniform(-1, 1, len(FEATURE_RANGES))
    target = features @ true_weights + np.random.normal(0, 0.5, n_samples)

    return np.column_stack([features, target])


class LinearRegression:
    def __init__(self, initial_data, collumn_headers): 
        #initial data is sxp matrix
        self.collumn_headers = collumn_headers

        self.num_params = len(initial_data[0]) #actually equal to num_params + 1 as last collumn is output
        self.num_samples = len(initial_data)

        self.a = np.array([np.append(row[:-1], 1) for row in initial_data])
        log(f"A:\n {self.a}", 2)

        self.at = self.a.transpose()
        log(f"AT:\n {self.at}", 2)

        self.b = initial_data[:, -1].reshape(self.num_samples,1)
        log(f"B:\n {self.b}", 2)

        self.ata = self.at @ self.a
        log(f"ata:\n {self.ata}", 2)

        self.ata_inv = np.linalg.inv(self.ata)
        log(f"ata inverse:\n {self.ata}", 2)

        self.atb = self.at @ self.b
        log(f"atb:\n {self.atb}", 2)

        self.params = self.ata_inv @ self.atb
        log(f"initial params:\n {self.params}", 1)
        self.print_equation()

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

    def stream_chunk_2(self, lines):
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
        log("recalculated params:", 2)

        log(f"\n {self.params}", 2)


if __name__ == "__main__":
    collumn_headers = list(FEATURE_RANGES.keys()) + ['1']

    input_data = generate_random_data(50)
    lr = LinearRegression(input_data, collumn_headers)

    append_line = generate_random_data(1).flatten()
    t1 = time.time()
    lr.stream_line(append_line)
    lr.recalculate_params()
    t2 = time.time()
    print(f"stream_line time: {t2-t1}")
    lr.print_equation()

    append_chunk = generate_random_data(1000)
    t3 = time.time()
    lr.stream_chunk(append_chunk)
    t4 = time.time()
    print(f"stream_chunk (1000 rows) time: {t4-t3}")
    lr.print_equation()



