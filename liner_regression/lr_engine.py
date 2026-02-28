import numpy as np
import time
from liner_regression.software_optimised_liner_regression import OptimisedSoftwareLR
from liner_regression.unoptimised_lr import OptimisedSoftwareLR, UnoptimisedSoftwareLR

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
    def __init__(self):
        collumn_headers = list(FEATURE_RANGES.keys()) + ['1']
        initial_input_data = generate_random_data(50)

        self.optimised_sw_lr =  self.initialise_optimised_sw_lr(initial_input_data, collumn_headers)
        self.unoptimised_sw_lr =  self.initialise_unoptimised_sw_lr(initial_input_data, collumn_headers)

    def initialise_optimised_sw_lr(self, data_in, collumn_headers):
        return OptimisedSoftwareLR(data_in, collumn_headers)

    def initialise_unoptimised_sw_lr(self, data_in, collumn_headers):
        return UnoptimisedSoftwareLR(data_in, collumn_headers)
    









