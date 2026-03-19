"""
Dummy script to POST feature names to the weights endpoint.
The feature list is shuffled randomly on each run.
"""

import json
import random
import urllib.request
import urllib.error

ENDPOINT = "http://13.60.162.169:5000/weights"

FEATURE_NAMES = [
    "spread_bps",
    "wmid_deviation",
    "book_imbalance",
    "book_slope_ratio",
    "depth_ratio",
    "vol_delta_ratio",
    "trade_intensity_z",
    "large_trade_ratio",
    "realized_volatility",
    "momentum_10s",
    "vwma_deviation_30s",
    "cum_volume_delta",
    "BIAS",
]


def build_payload(feature_names: list[str]) -> dict:
    shuffled = feature_names[:]
    random.shuffle(shuffled)
    return {"asset": "BTC", "features": shuffled}


def post_payload(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    payload = build_payload(FEATURE_NAMES)
    print("Sending payload:")
    print(json.dumps(payload, indent=2))
    print()

    try:
        result = post_payload(ENDPOINT, payload)
        print("Response:")
        print(json.dumps(result, indent=2))
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")


if __name__ == "__main__":
    main()
