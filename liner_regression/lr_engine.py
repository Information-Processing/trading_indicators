import numpy as np
import time
from liner_regression.hardware_lr import HardwareLinearRegression
from software_optimised_liner_regression import OptimisedSoftwareLR
from unoptimised_lr import UnoptimisedSoftwareLR

# example input:
# [[1,2,3,4,5],
#  [1,2,3,4,5]]

FEATURE_RANGES = {
    "VWAP": (95.0, 105.0),
    "Vol": (1.0, 10.0),
    "BookImb": (0.3, 0.75),
    "Spread": (0.05, 0.25),
    "BuySellRatio": (0.4, 2.5),
    "LogRet": (-0.01, 0.01),
    "TradeRate": (20.0, 70.0),
    "Volatility": (0.03, 0.25),
    "CVD": (-20.0, 45.0),
    "BidDepth": (50.0, 500.0),
    "AskDepth": (50.0, 500.0),
    "PriceRange": (0.01, 2.0),
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

class LinearRegressionEngine:
    def __init__(self, ip):
        collumn_headers = list(FEATURE_RANGES.keys()) + ['1']
        initial_input_data = generate_random_data(50)

        self.unoptimised_sw_lr =  self.initialise_unoptimised_sw_lr(initial_input_data, collumn_headers)
        self.optimised_sw_lr =  self.initialise_optimised_sw_lr(initial_input_data, collumn_headers)
        self.hardware_lr =  self.initialise_hardware_lr(ip, collumn_headers)

    def initialise_optimised_sw_lr(self, data_in, collumn_headers):
        return OptimisedSoftwareLR(data_in, collumn_headers)

    def initialise_unoptimised_sw_lr(self, data_in, collumn_headers):
        return UnoptimisedSoftwareLR(data_in, collumn_headers)

    def initialise_hardware_lr(self, ip, collumn_headers):
        return HardwareLinearRegression(ip, collumn_headers)

    def test_unoptimised_sw_lr(self, samples):
        self.unoptimised_sw_lr.stream_chunk(samples)

    def test_optimised_sw_lr(self, samples):
        self.optimised_sw_lr.stream_chunk_optimised(samples)
    
    def test_hardware_lr(self, samples):
        self.optimised_sw_lr.stream_chunk(samples)

    def test_all_lr(self, num_samples):
        samples = generate_random_data(num_samples)
        print(f"TESTING {num_samples} samples:\n\n")

        t1 = time.time()
        self.test_unoptimised_sw_lr(samples)
        t2 = time.time()
        self.test_optimised_sw_lr(samples)
        t3 = time.time()
        self.test_hardware_lr(samples)
        t4 = time.time()

        print("time for unoptimised software:") 
        print(f"{t2-t1}\n")
        print("time for optimsed software:") 
        print(f"{t3-t2}\n")
        print("time for optimsed hardware:") 
        print(f"{t4-t3}\n")

    def print_all_equations(self):
        print(f"\nunoptimised software equation:")
        self.unoptimised_sw_lr.print_equation()

        print(f"\noptimised software equation:")
        self.optimised_sw_lr.print_equation()

        print(f"\noptimised software equation:")
        self.hardware_lr.print_equation()

if __name__ == "__main__":
    IP = 1
    lr_engine = LinearRegressionEngine(IP)
    
    lr_engine.test_all_lr(1000)
    lr_engine.print_all_equations()





