import numpy as np
import time

# example input:
# [[1,2,3,4,5],
#  [1,2,3,4,5]]

LOG_LVL = 1

def log(message, level):
    if (level <= LOG_LVL):
        print(message)


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
        if len(line) != self.num_params:
            log("ERROR: line and param size diff", 0)
            return
        
        line_output = line[-1]
        q_vec = np.append(line[:-1], 1).reshape(1, self.num_params)
        qt_vec = q_vec.reshape(self.num_params, 1)

        log(f"q:\n {q_vec}", 2)
        log(f"qt:\n {qt_vec}", 2)

        ata_diff = qt_vec @ q_vec
        atb_diff = line_output * qt_vec

        log(f"ata diff:\n {ata_diff}", 2)
        log(f"atb diff:\n {atb_diff}", 2)

        self.ata += ata_diff
        self.atb += atb_diff

    def stream_chunk(self, lines):
        for line in lines:
            self.stream_line(np.array(line))
        self.recalculate_params()

    def recalculate_params(self):
        self.ata_inv = np.linalg.inv(self.ata)
        self.params = self.ata_inv @ self.atb
        log("recalculated params:", 2)

        log(f"\n {self.params}", 2)


if __name__ == "__main__":
    collumn_headers = ['VWAP', 'Vol', 'BookImb', 'Spread', 'BuySellRatio', 'LogRet', 'TradeRate', 'Volatility', 'CVD', '1']
    input_data = np.array([
        [100.0, 5.0, 0.55, 0.12, 1.3, 0.002,  45.0, 0.08, 10.5,  101.2],
        [101.5, 3.2, 0.48, 0.15, 0.9, -0.001, 38.0, 0.12, -3.2,  102.0],
        [99.8,  7.1, 0.62, 0.10, 1.7, 0.005,  52.0, 0.15, 22.1,  100.5],
        [102.3, 2.5, 0.41, 0.18, 0.7, -0.003, 30.0, 0.06, -8.4,  103.1],
        [98.5,  8.0, 0.70, 0.09, 2.1, 0.008,  60.0, 0.20, 31.0,   99.0],
        [101.0, 4.8, 0.52, 0.14, 1.1, 0.001,  42.0, 0.10,  5.3,  101.8],
        [100.2, 6.3, 0.58, 0.11, 1.5, 0.003,  48.0, 0.13, 15.7,  100.9],
        [103.0, 2.0, 0.38, 0.20, 0.6, -0.004, 28.0, 0.05, -12.0, 103.5],
        [99.0,  9.1, 0.72, 0.08, 2.3, 0.010,  65.0, 0.22, 40.2,   99.5],
        [101.8, 3.8, 0.50, 0.13, 1.0, 0.000,  40.0, 0.09,  1.0,  102.2],
    ])

    lr = LinearRegression(input_data, collumn_headers)

    append_line = np.array([102.5, 4.0, 0.53, 0.12, 1.2, 0.002, 44.0, 0.11, 7.8, 103.0])

    t1 = time.time()

    lr.stream_line(append_line)
    lr.recalculate_params()

    t2 = time.time()
    print(f"time diff: {t2-t1}")

    lr.print_equation()

    append_chunk = np.array([
        [100.5, 5.5, 0.56, 0.11, 1.4, 0.003,  46.0, 0.09, 12.0, 101.0],
        [101.2, 3.0, 0.47, 0.16, 0.8, -0.002, 36.0, 0.11, -5.0, 102.5],
        [99.5,  7.5, 0.64, 0.09, 1.8, 0.006,  55.0, 0.16, 25.3, 100.0],
        [102.0, 2.8, 0.43, 0.17, 0.8, -0.003, 32.0, 0.07, -6.1, 102.8],
        [98.8,  8.5, 0.68, 0.08, 2.0, 0.009,  62.0, 0.19, 35.5,  99.2],
        [101.3, 4.2, 0.51, 0.13, 1.0, 0.001,  41.0, 0.10,  3.8, 101.5],
        [100.8, 6.0, 0.59, 0.10, 1.6, 0.004,  50.0, 0.14, 18.0, 101.2],
        [103.2, 1.8, 0.36, 0.21, 0.5, -0.005, 26.0, 0.04, -15.0, 103.8],
        [99.2,  8.8, 0.71, 0.07, 2.2, 0.011,  63.0, 0.21, 38.0,  99.8],
        [102.0, 3.5, 0.49, 0.14, 1.1, 0.001,  39.0, 0.08,  2.5, 102.4],
    ])

    t3 = time.time()

    lr.stream_chunk(append_chunk)

    t4 = time.time()
    print(f"time diff: {t4-t3}")
    lr.print_equation()



