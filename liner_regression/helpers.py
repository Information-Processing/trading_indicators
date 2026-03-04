import numpy as np

FEATURE_RANGES = {
    "VWAP": (-1.0, 1.0),
    "Vol": (-1.0, 1.0),
    "BookImb": (-1.0, 1.0),
    "Spread": (-1.0, 1.0),
    "BuySellRatio": (-1.0, 1.0),
    "LogRet": (-1.0, 1.0),
    "TradeRate": (-1.0, 1.0),
    "Volatility": (-1.0, 1.0),
    "CVD": (-1.0, 1.0),
    "BidDepth": (-1.0, 1.0),
    "AskDepth": (-1.0, 1.0),
    "PriceRange": (-1.0, 1.0),
}


def generate_random_data(n_samples):
    cols = []
    for bounds in FEATURE_RANGES.values():
        cols.append(np.random.uniform(bounds[0], bounds[1], n_samples))

    features = np.column_stack(cols)
    #make the target (yval) a distribution of the inputs so its not completely rng
    true_weights = np.random.uniform(-1, 1, len(FEATURE_RANGES))
    target = np.clip(
        features @ true_weights + np.random.normal(0, 0.1, n_samples), -1.0, 1.0
    )

    return np.column_stack([features, target])

