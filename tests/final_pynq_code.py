
API_BASE_URL = "http://13.60.162.169:5000"

#This is the global map of features were initially trading off
FEATURE_RANGES = {
    "spread_bps":           0,
    "wmid_deviation":       0,
    "book_imbalance":       0,
    "book_slope_ratio":     0,
    "depth_ratio":          0,
    "vol_delta_ratio":      0,
    "trade_intensity_z":    0,
    "large_trade_ratio":    0,
    "realized_volatility":  0,
    "momentum_10s":         0,
    "vwma_deviation_30s":   0,
    "cum_volume_delta":     0,
    "last_price":           0
}

# update map of features that we will be tradding off
# get map off flask server /weights endpoint
def update_feature_ranges(eng):
    global FEATURE_RANGES

    response = requests.get(f"{API_BASE_URL}/weights").json()
    new_features = [x for x in response["features"] if x != "BIAS"]
    new_features.append("last_price")

    old_features = list(FEATURE_RANGES.keys())

    if new_features != old_features:
        FEATURE_RANGES = {x: 0 for x in new_features}
        eng.lr_engine = LinearRegressionEngine(ip)
        eng.ret_dict.clear()
        eng.price_history.clear()
        print("\nNew paramaters detected\n")


# helper function to convert dictionary to numpy
# used in LREngine for effiecient loads to FPGA
def bundle_dict_to_numpy(data_dict):
    keys = list(FEATURE_RANGES.keys())
    min_len = min(len(data_dict[k]) for k in keys)
    cols = [data_dict[k][:min_len] for k in keys]
    return np.column_stack(cols).astype(np.float64)

# float normaliser, class to z normalise inputs before we put it into LREngine
# industry standard to let the model focus on patterns in the data rather than scale
class FloatNormaliser:
    def __init__(self, num_features):
        self.num_features = num_features
        self.n = 0
        self._mean = np.zeros(num_features)
        self._mom2 = np.zeros(num_features)
   
    # update second moment and mean per batch. scaled relative to batch size
    def _update_stats(self, batch):
        batch_n = batch.shape[0]
        if batch_n == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_mom2 = batch.var(axis=0, ddof=0) * batch_n
        if self.n == 0:
            self._mean = batch_mean
            self._mom2 = batch_mom2
        else:
            total_n = self.n + batch_n
            delta = batch_mean - self._mean
            self._mean += delta * (batch_n / total_n)
            self._mom2 += batch_mom2 + delta ** 2 * (self.n * batch_n / total_n)
        self.n += batch_n
   
    # return sample standard deviation
    def std(self):
        if self.n < 2:
            return np.ones(self.num_features)
       
        standard_dev = np.sqrt(self._mom2 / (self.n - 1))
        standard_dev[standard_dev < 1e-10] = 1.0 #replace low standard deviation with 1
        return standard_dev

    def reset(self):
        self.n = 0
        self._mean = np.zeros(self.num_features)
        self._mom2 = np.zeros(self.num_features)
   
    # z normalise between -3 and 3. returns float64
    def normalise(self, samples):
        self._update_stats(samples)
        z = (samples - self._mean) / self.std()
        return np.clip(z, -3.0, 3.0)

    # takes in weights between -3 to 3 and scales them back to true form
    # np multiply by sigma and add mean for all, bias dealt with seperately
    def denormalise_weights(self, weights_norm):
        orig_shape = weights_norm.shape
        w = weights_norm.flatten()

        feat_std = self.std()[:-1]
        target_std = self.std()[-1]
        feat_mean = self._mean[:-1]
        target_mean = self._mean[-1]

        w_feat = w[:-1] * (target_std / feat_std)
        w_bias = w[-1] * target_std + target_mean - np.sum(w_feat * feat_mean)

        return np.concatenate([w_feat, [w_bias]]).reshape(orig_shape)


# hardware linear regression model
class HardwareLR:
    #dimentions
    D = 13
    NUM_WORDS = 16

    ACC_FRAC_BITS = 24
    HW_BATCH_SIZE = 4096
   
    #addresses
    ADDR_AP_CTRL     = 0x000
    ADDR_MEM_IN_DATA = 0x010
    ADDR_NUM_SAMPLES = 0x01c
    ADDR_ATB_BASE    = 0x080
    ADDR_ATA_BASE    = 0x800

    def __init__(self, ip, column_headers, max_samples=32768):
        self.ip = ip
        self.column_headers = column_headers
        self.num_params = len(column_headers)
        self.weights = np.zeros((self.D, 1))
        self.max_samples = max_samples


        # pynq allocate addresses for writing samples into
        self._mem_in = allocate(shape=(max_samples, self.NUM_WORDS), dtype=np.int32)
       
        #register addresses top and bottom of allocated memory
        self._mem_addr_lo = int(self._mem_in.device_address) & 0xFFFFFFFF
        self._mem_addr_hi = (int(self._mem_in.device_address) >> 32) & 0xFFFFFFFF
       
        #initialise ata and atb matricies for rank one update calcs
        self.ata = np.zeros((self.num_params, self.num_params), dtype=np.float64)
        self.atb = np.zeros((self.num_params, 1), dtype=np.float64)

    # reset engine for new assets or features
    def reset(self):
        self.weights = np.zeros((self.D, 1))
        self.ata = np.zeros((self.num_params, self.num_params), dtype=np.float64)
        self.atb = np.zeros((self.num_params, 1), dtype=np.float64)
   
    #convert array from floats to bits
    def _floats_to_bits(self, arr):
        return np.ascontiguousarray(arr, dtype=np.float32).view(np.int32)

   
    def _run_hw_batch(self, x_float, y_float):
        n = len(y_float)
       
        # fill first row with outputs
        self._mem_in[:n, 0] = self._floats_to_bits(y_float)
        # fill all remaining rows as features
        self._mem_in[:n, 1:self.D + 1] = self._floats_to_bits(x_float)
        self._mem_in.flush()
       
        # write number of samples and start flags
        # using np instead of for loop to reduce pythonic overhead
        self.ip.write(self.ADDR_NUM_SAMPLES, n)
        self.ip.write(self.ADDR_MEM_IN_DATA, self._mem_addr_lo)
        self.ip.write(self.ADDR_MEM_IN_DATA + 4, self._mem_addr_hi)
        self.ip.write(self.ADDR_AP_CTRL, 0x01)

        # spin until done
        while not (self.ip.read(self.ADDR_AP_CTRL) & 0x02):
            time.sleep(0.000001)

        # read results and increment rank one update matricies based on them
        hw_ata, hw_atb = self._read_hw_results()

        self.ata += hw_ata
        self.atb += hw_atb

    # reads 64 bit array count times starting at base_addr and outputs a float64 array
    def _read_64bit_array(self, base_addr, count):
        word_start = base_addr // 4
        raw = np.array(
            self.ip.mmio.array[word_start : word_start + count * 2],
            dtype=np.uint32,
        )
        lo = raw[0::2].astype(np.int64)
        hi = raw[1::2].astype(np.int64)
        combined = (hi << 32) | lo   # bit reconstruction
        return combined.astype(np.float64) / (1 << self.ACC_FRAC_BITS)

    # reads ata atb registers
    def _read_hw_results(self):
        hw_ata = self._read_64bit_array(
            self.ADDR_ATA_BASE, self.D * self.D
        ).reshape(self.D, self.D)

        hw_atb = self._read_64bit_array(
            self.ADDR_ATB_BASE, self.D
        ).reshape(self.D, 1)

        return hw_ata, hw_atb

    # runs a batch of samples at max 4096 a time and updates weights
    def stream_chunk(self, samples_norm):
        test_y = samples_norm[:, -1].astype(np.float32)
        test_x = np.concatenate([
            samples_norm[:, :-1].astype(np.float32),
            np.ones((samples_norm.shape[0], 1), dtype=np.float32),
        ], axis=1)

        n = len(test_y)
        for start in range(0, n, self.HW_BATCH_SIZE):
            end = min(start + self.HW_BATCH_SIZE, n)
            self._run_hw_batch(test_x[start:end], test_y[start:end])

        self.weights = np.linalg.pinv(self.ata) @ self.atb


class LinearRegressionEngine:
    def __init__(self, ip):
        feat_names = list(FEATURE_RANGES.keys())[:-1]
        self.column_headers = feat_names + ["BIAS"]
        self.normaliser = FloatNormaliser(len(FEATURE_RANGES))
        self.hardware_lr = HardwareLR(ip, self.column_headers, max_samples=32768)
       
    def _get_denormed(self, weights_norm):
        return self.normaliser.denormalise_weights(weights_norm).flatten().tolist()

    def reset(self):
        self.normaliser.reset()
        self.hardware_lr.reset()

    # converts dictionary to numpy, runs hardware batch and times it
    def test_all_lr(self, ret_dict):
        samples_float = bundle_dict_to_numpy(ret_dict)
        samples_norm = self.normaliser.normalise(samples_float)

       
        t1 = time.time()
        self.hardware_lr.stream_chunk(samples_norm)
        t2 = time.time()

        print(f"Samples: {len(samples_norm)} | Hardware Time: {t2-t1:.4f}s")

    def _print_denormed(self, weights_norm):
        denormed = self.normaliser.denormalise_weights(weights_norm)
        weights = denormed.flatten()
        print(f"{weights[0]:.6g}*{self.column_headers[0]}", end = "")
        for i in range(1, len(weights) - 1):
            print(f" + {weights[i]:.6g}*{self.column_headers[i]}", end="")
        print(f" + {weights[-1]:.6g}")

    def print_all_equations(self):
        print("\n[HARDWARE FPGA WEIGHTS]")
        self._print_denormed(self.hardware_lr.weights)
   
    # post weights to flask server
    def post_weights(self, asset):
        weights = self._get_denormed(self.hardware_lr.weights)
        features = self.column_headers
        payload = {
            "weights": weights,
            "features": features,
            "asset": asset,
        }

        resp = requests.post(f"{API_BASE_URL}/matrix", json=payload)

    def get_weights(self):
        return self._get_denormed(self.hardware_lr.weights)

POLLING_PERIOD = 0.2
WARMUP_PERIOD = 5
WARMUP_ITERATIONS = WARMUP_PERIOD / POLLING_PERIOD


class Engine:
    def __init__(self, ip, binance_ws):
        self.binance_ws = binance_ws
        self.ce = CalculationEngineV2()
       
        self.ret_dict = defaultdict(list)
        self.lr_engine = LinearRegressionEngine(ip)
       
        self.price_history = deque()
       
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()
   
    def reset(self):
        self.lr_engine.reset()
        self.price_history = deque()
       
    def stop(self):
        self.stop_event.set()
   
    def get_data(self):
        iterations = 0
        trades = self.binance_ws.trades
        last_price = self.binance_ws.last_price

        while not self.stop_event.is_set():
            now = time.time()
            trades_snapshot = list(self.binance_ws.trades)
            trades_10 = self.binance_ws.get_trades_since(now - 11.0)
            trades_30 = self.binance_ws.get_trades_since(now - 31.0)
            shortterm_trades = self.binance_ws.get_trades_since(now - 1)

            order_book = self.binance_ws.order_book
            asks = order_book.get("asks", [])
            bids = order_book.get("bids", [])

            if not asks or not bids:
                if self.stop_event.wait(1):
                    break
                continue

            best_ask = asks[0]
            best_bid = bids[0]
            last_price = self.binance_ws.last_price
           
            # calculate all features
            spread      = self.ce.spread_bps(best_ask[0], best_bid[0])
            wmid_dev    = self.ce.weighted_mid_deviation(best_ask, best_bid)
            imbalance   = self.ce.book_imbalance(asks, bids, levels=10)
            slope_ratio = self.ce.book_slope_ratio(asks, bids, levels=5)
            depth_r     = self.ce.depth_ratio(asks, bids)
            vol_delta   = self.ce.volume_delta_ratio(trades_snapshot, now, 5.0)
            intensity   = self.ce.trade_intensity_zscore(len(shortterm_trades))
            large_ratio = self.ce.large_trade_ratio(trades_snapshot, now, 5.0, threshold_qty=0.1)
            volatility  = self.ce.realized_volatility(trades_10, now, 10.0)
            mom         = self.ce.momentum(trades_10, now, 10.0)
            vwma_dev    = self.ce.vwma_deviation(trades_30, now, 30.0, last_price)
            cum_vdelta  = self.ce.cumulative_volume_delta(vol_delta)
            accel       = self.ce.price_acceleration(trades_snapshot, now)
            bs_ratio    = self.ce.buy_sell_trade_count_ratio(trades_snapshot, now)
            sz_zscore   = self.ce.trade_size_zscore(trades_snapshot, now)
            price_range = self.ce.price_range_bps(trades_10, now, 10.0)
            tick_dir    = self.ce.tick_direction_ratio(trades_snapshot, now)
            vwap_dev    = self.ce.vwap_deviation_short(trades_snapshot, now, 5.0, last_price)
            ask_press   = self.ce.ask_touch_pressure(asks)
            bid_press   = self.ce.bid_touch_pressure(bids)
            roll_spr    = self.ce.roll_spread_estimate(trades_10, now, 10.0)
            mid_retz    = self.ce.mid_price_returns_zscore(asks, bids)
            sz_disp     = self.ce.trade_size_dispersion(trades_snapshot, now)
            iti_zscore  = self.ce.inter_trade_interval_zscore(trades_snapshot, now)
           
            # append to return dict with data lock: we dont want LREngine to be reading from ret_dict while appending to it
            with self.data_lock:
                self.ret_dict["spread_bps"].append(spread)
                self.ret_dict["wmid_deviation"].append(wmid_dev)
                self.ret_dict["book_imbalance"].append(imbalance)
                self.ret_dict["book_slope_ratio"].append(slope_ratio)
                self.ret_dict["depth_ratio"].append(depth_r)
                self.ret_dict["vol_delta_ratio"].append(vol_delta)
                self.ret_dict["trade_intensity_z"].append(intensity)
                self.ret_dict["large_trade_ratio"].append(large_ratio)
                self.ret_dict["realized_volatility"].append(volatility)
                self.ret_dict["momentum_10s"].append(mom)
                self.ret_dict["vwma_deviation_30s"].append(vwma_dev)
                self.ret_dict["cum_volume_delta"].append(cum_vdelta)
                self.ret_dict["price_acceleration"].append(accel)
                self.ret_dict["buy_sell_count_ratio"].append(bs_ratio)
                self.ret_dict["trade_size_zscore"].append(sz_zscore)
                self.ret_dict["price_range_bps"].append(price_range)
                self.ret_dict["tick_direction_ratio"].append(tick_dir)
                self.ret_dict["vwap_deviation_short"].append(vwap_dev)
                self.ret_dict["ask_touch_pressure"].append(ask_press)
                self.ret_dict["bid_touch_pressure"].append(bid_press)
                self.ret_dict["roll_spread_estimate"].append(roll_spr)
                self.ret_dict["mid_returns_zscore"].append(mid_retz)
                self.ret_dict["trade_size_dispersion"].append(sz_disp)
                self.ret_dict["inter_trade_interval_z"].append(iti_zscore)
               
                # live queue of size HORIZON_STEPS. buffer to predict future log return
                # log return used to scale how much we think it will go up or down
                if iterations > WARMUP_ITERATIONS:
                    self.price_history.append(last_price)

                    if len(self.price_history) > HORIZON_STEPS + 1:
                        self.price_history.popleft()

                    if len(self.price_history) == HORIZON_STEPS + 1:
                        p_t = self.price_history[0]
                        p_tH = self.price_history[-1]

                        log_return = float(np.log(p_tH / p_t))

                        self.ret_dict["last_price"].append(log_return)
                       
            iterations += 1

            if self.stop_event.wait(POLLING_PERIOD):
                break



class TestingEngine:
    def __init__(self, binance_ws):
        self.binance_ws = binance_ws
        self.ce = CalculationEngineV2()
        self.prev_price = None
        self.prev_model_signal = None

        self.model_correct_predictions = 0
        self.model_incorrect_predictions = 0

        self.p_price_accuracy = 0.0
       
        #flag to see if we changed the asset we are trading e.g BTC to ETH
        self.changed_asset = True
       
        #initial fake balance
        self.fake_money = 1000.0
       
        self.leverage = 20
   
    # runs the calculation engine to compute all features for testing
    def _build_feature_vector(self, now, order_book, trades_snapshot, trades_10, trades_30, shortterm_trades, last_price):
        asks = order_book.get("asks", [])
        bids = order_book.get("bids", [])

        if not asks or not bids or last_price is None:
            return None

        best_ask = asks[0]
        best_bid = bids[0]

        spread = self.ce.spread_bps(best_ask[0], best_bid[0])
        wmid_dev = self.ce.weighted_mid_deviation(best_ask, best_bid)
        imbalance = self.ce.book_imbalance(asks, bids, levels=10)
        slope_ratio = self.ce.book_slope_ratio(asks, bids, levels=5)
        depth_r = self.ce.depth_ratio(asks, bids)
        vol_delta = self.ce.volume_delta_ratio(trades_snapshot, now, 5.0)
        intensity = self.ce.trade_intensity_zscore(len(shortterm_trades))
        large_ratio = self.ce.large_trade_ratio(trades_snapshot, now, 5.0, threshold_qty=0.1)
        volatility = self.ce.realized_volatility(trades_10, now, 10.0)
        mom = self.ce.momentum(trades_10, now, 10.0)
        vwma_dev = self.ce.vwma_deviation(trades_30, now, 30.0, last_price)
        cum_vdelta = self.ce.cumulative_volume_delta(vol_delta)
        accel       = self.ce.price_acceleration(trades_snapshot, now)
        bs_ratio    = self.ce.buy_sell_trade_count_ratio(trades_snapshot, now)
        sz_zscore   = self.ce.trade_size_zscore(trades_snapshot, now)
        price_range = self.ce.price_range_bps(trades_10, now, 10.0)
        tick_dir    = self.ce.tick_direction_ratio(trades_snapshot, now)
        vwap_dev    = self.ce.vwap_deviation_short(trades_snapshot, now, 5.0, last_price)
        ask_press   = self.ce.ask_touch_pressure(asks)
        bid_press   = self.ce.bid_touch_pressure(bids)
        roll_spr    = self.ce.roll_spread_estimate(trades_10, now, 10.0)
        mid_retz    = self.ce.mid_price_returns_zscore(asks, bids)
        sz_disp     = self.ce.trade_size_dispersion(trades_snapshot, now)
        iti_zscore  = self.ce.inter_trade_interval_zscore(trades_snapshot, now)
       
        feature_map = {}
        feature_map["spread_bps"] = spread
        feature_map["wmid_deviation"] = wmid_dev
        feature_map["book_imbalance"] = imbalance
        feature_map["book_slope_ratio"] = slope_ratio
        feature_map["depth_ratio"] = depth_r
        feature_map["vol_delta_ratio"] = vol_delta
        feature_map["trade_intensity_z"] = intensity
        feature_map["large_trade_ratio"] = large_ratio
        feature_map["realized_volatility"] = volatility
        feature_map["momentum_10s"] = mom
        feature_map["vwma_deviation_30s"] = vwma_dev
        feature_map["cum_volume_delta"] = cum_vdelta
        feature_map["price_acceleration"] = accel
        feature_map["buy_sell_count_ratio"] = bs_ratio
        feature_map["trade_size_zscore"] = sz_zscore
        feature_map["price_range_bps"] = price_range
        feature_map["tick_direction_ratio"] = tick_dir
        feature_map["vwap_deviation_short"] = vwap_dev
        feature_map["ask_touch_pressure"] = ask_press
        feature_map["bid_touch_pressure"] = bid_press
        feature_map["roll_spread_estimate"] = roll_spr
        feature_map["mid_returns_zscore"] = mid_retz
        feature_map["trade_size_dispersion"] = sz_disp
        feature_map["inter_trade_interval_z"] = iti_zscore
       
        # only reads the relevant features into the returned array
        ret_arr = []
        for feature in FEATURE_RANGES.keys():
            if feature != "last_price":
                ret_arr.append(feature_map[feature])
   
        return np.array(ret_arr, dtype=np.float64)

    # balance helper
    def _signal_from_prediction(self, predicted_return, threshold: float = 0.0):
        if predicted_return > threshold:
            return "BUY"
        else:
            return "SELL"
       
    def post_acc_bal(self):
        post_data = {"accuracy":self.p_price_accuracy,"balance":self.fake_money}
        requests.post(f"{API_BASE_URL}/portfolio", json = post_data)
       
       
    # actually uses the weights from hardware to compute prediction on features
    def use_weights(self, weights):
        now = time.time()
        trades_snapshot = list(self.binance_ws.trades)
        trades_10 = self.binance_ws.get_trades_since(now - 11.0)
        trades_30 = self.binance_ws.get_trades_since(now - 31.0)
        shortterm_trades = self.binance_ws.get_trades_since(now - 1.0)

        order_book = self.binance_ws.order_book
        last_price = self.binance_ws.last_price

        features = self._build_feature_vector(
            now,
            order_book,
            trades_snapshot,
            trades_10,
            trades_30,
            shortterm_trades,
            last_price,
        )

        weights_arr = np.asarray(weights, dtype=np.float64)
       
        #predict and give log return
        pred = float(np.dot(weights_arr[:-1], features) + weights_arr[-1])

        #do price to price prediction
        actual_signal = "N/A"
        if self.prev_price is not None and self.prev_model_signal is not None:
            #compute correct signal
            price_moved_up = last_price > self.prev_price
            actual_signal = "BUY" if price_moved_up else "SELL"
           
            #count correct and incorrect predictions
            if self.prev_model_signal == actual_signal:
                self.model_correct_predictions += 1
            else:
                self.model_incorrect_predictions += 1

            #computing new fake balance
            price_ratio = last_price / self.prev_price if self.prev_price != 0 else 1.0
            pct_change = price_ratio - 1.0

            if not self.changed_asset:
                if self.prev_model_signal == "BUY":
                    equity_multiplier = 1.0 + self.leverage * pct_change
                else:
                    equity_multiplier = 1.0 - self.leverage * pct_change

                equity_multiplier = max(0.0, equity_multiplier)  # optional liquidation floor
                self.fake_money *= equity_multiplier
            else:
                self.changed_asset = False

        # generate new signals for the next interval using predicted return
        model_signal = self._signal_from_prediction(pred)

        self.prev_price = last_price
        self.prev_model_signal = model_signal

        model_total = self.model_correct_predictions + self.model_incorrect_predictions
       
        if model_total != 0:
            self.p_price_accuracy = (self.model_correct_predictions / model_total)
        else:
            self.p_price_accuracy = 1
       

        print("=" * 100)
        print(f"num iterations: {model_total}\n")
        print(f"prediction: {pred}, actual price: {last_price}\n")
        print(f"p-p signal: {model_signal}, actual signal: {actual_signal}\n")
        print(f"p-p accuracy: {self.p_price_accuracy * 100}%\n")
        print(f"BALANCE: {self.fake_money}\n")
        print("=" * 100)

if __name__ == "__main__":
    REQUIRED_LABELS = 15
    HORIZON_STEPS = 10
    HORIZON_SECONDS = HORIZON_STEPS * POLLING_PERIOD

    print(f"running with samples={REQUIRED_LABELS}, horizon_seconds={HORIZON_SECONDS}")
   
    binance_ws_client = BinanceWSClient()
    binance_ws_client.run_ws()
   
    eng = Engine(ip, binance_ws_client)
    #run get_data on a daemon thread
    data_thread = threading.Thread(target=eng.get_data, daemon=True)
    data_thread.start()
   
    test_binance = TestingEngine(binance_ws_client)

    current_asset = "BTC"
    asset_lower = current_asset.lower()

    try:
        while True:
            time.sleep(HORIZON_SECONDS)
            data_copy = None
            num_labels = 0
            with eng.data_lock:
                num_labels = len(eng.ret_dict["last_price"])
               
                #wait for input buffer to be filled in the eng.ret_dict
                if num_labels >= REQUIRED_LABELS:
                    data_copy = {k: list(v[:num_labels]) for k, v in eng.ret_dict.items()}
                    eng.ret_dict.clear()
                    eng.price_history.clear()
                    weights = eng.lr_engine.get_weights()
                    test_binance.use_weights(weights)
                    test_binance.post_acc_bal()
                   
            #run linear regression
            if data_copy is not None:
                eng.lr_engine.test_all_lr(data_copy)
                eng.lr_engine.print_all_equations()
                eng.lr_engine.post_weights(asset_lower)
            else:
                print(f"Warming up labels... {num_labels}/{REQUIRED_LABELS}")
               
            # check for changed asset
            response = requests.get(f"{API_BASE_URL}/instruction").json()
            fetched = response["asset"]
            if fetched != current_asset:
                print(f"Switching from {current_asset} to {fetched}")
                current_asset = fetched
                binance_ws_client.update_url(current_asset)
                test_binance.changed_asset = True
                eng.reset()
                data_copy = None
               
            #check for new feature set
            update_feature_ranges(eng)
            asset_lower = current_asset.lower()
                   
               
    except KeyboardInterrupt:
        print("Shutting down...")
        eng.stop()
        binance_ws_client.stop()
        data_thread.join(timeout=5)rom collections import deque
