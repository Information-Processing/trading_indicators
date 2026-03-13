# Generated from: experimental porject.ipynb
# Converted at: 2026-03-13T00:38:42.371Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================================================================
# BLOCK 0: IMPORTS 
# ============================================================================

import numpy as np
import speech_recognition as sr
from os import system
from gtts import gTTS
import tempfile

import logging
from enum import Enum
import time

from pynq import Overlay

import json
import websocket
from enum import Enum
import threading

from cffi import FFI

import requests
import new_audio
from scipy import signal

# ============================================================================
# BLOCK 1: GLOBALS 
# ============================================================================
BASE_URL = "http://13.60.162.169:5000"
INSTRUCTION_ENDPOINT = f"{BASE_URL}/store_instruction"
OPENAI_API_KEY = "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"

#declare exponential moving avg
agc_running_peak = None
#exp moving avg alpha = smoothing factor
AGC_ALPHA = 0.3
#target frequency for normalise pcm
TARGET_FS = 16000

#Listener Thread config
WEIGHTS_ENDPOINT = f"{BASE_URL}/weights"
LISTENER_INTERVAL_SEC = 1 # Look for new data every second
active_asset = "BTC" # updated by the voice assistant, switches between ETH/BTC

# ============================================================================
# BLOCK 2: LOGGING
# ============================================================================

#logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

#suppress noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('gtts').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


# ============================================================================
# BLOCK 3: PYNQ INIT
# ============================================================================

import os
_overlay_dir = os.getcwd()
_overlay_path = os.path.join(_overlay_dir, 'base.bit')
overlay = Overlay(_overlay_path)
logger.info(f"overlay loaded with keys: {overlay.ip_dict.keys()}")

pynq_audio = overlay.audio_direct_v1_1_0
pynq_audio._ffi = FFI()
logger.info(f"driver type (must be DirectAudio class): {type(pynq_audio)}")


# ============================================================================
# BLOCK 4: PCM Cleanup
# ============================================================================
def normalized_pcm(audio, samples=None):
    global agc_running_peak

    #get raw audio capped at sample len, flattend and as float 32 bit
    raw = audio.buffer[:audio.sample_len].flatten().astype(np.float32)

    up = int(TARGET_FS) #upsample factor (what we want)
    down = int(audio.sample_rate) #downsample factor (what we have)
    audio_data = signal.resample_poly(raw, up, down) #downsample with anti aliasing

    # remove dc signal = avg
    audio_data -= np.mean(audio_data)

    current_peak = np.max(np.abs(audio_data)) #get maximum

    #initialise agc_running_peak on first call
    if agc_running_peak is None:
        agc_running_peak = current_peak
    #else use exp moving average
    else: 
        agc_running_peak = AGC_ALPHA * current_peak + (1 - AGC_ALPHA) * agc_running_peak

    #calculate gain by using exp moving average and maximum value of integer
    gain = np.iinfo(np.int16).max / max(1e-7, agc_running_peak)
    #shove audio data * gain into int 16 min and max values
    audio_data = np.clip(audio_data * gain, np.iinfo(np.int16).min, np.iinfo(np.int16).max)

    #return as 16khz signal
    return audio_data.astype(np.int16)


# ============================================================================
# BLOCK 5: GTTS Class
# ============================================================================

class GttsCli:
    def say(self, text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_pcm:
            try:
                # sending text to gtts and sending that to mp3 temp file
                t0 = time.time()
                tts = gTTS(text)
                tts.write_to_fp(mp3)
                mp3.flush()
                
                
                # convert mp3 to wav 16k pcm signal at 8kHz and pass with high pass filter at 500Hz 
                # CHECK IF THIS FILTER IS FINE
                cmd = (
                    f"ffmpeg -loglevel error -y -i {mp3.name} "
                    f"-c:a pcm_s16le -ac 1 -ar 8000 "
                    f"-af 'highpass=f=500,volume=0.8' {wav_pcm.name}"
                )
                system(cmd)
                t1 = time.time()
                
                pynq_audio.load(wav_pcm.name)
                pynq_audio.play()
                t2 = time.time()
                
                logger.info(f"[Timing] GTTS PCM Generation: {t1-t0:.1f}s, PCM -> PDM + play: {t2-t1:.1f}")
                

            except Exception as e:
                logger.error(f"GTTS say error: {e}")


# ============================================================================
# BLOCK 6: OPENAI WEBSOCKET (CLASSIFIER: SWITCH TO ETHEREUM / BITCOIN)
# ============================================================================
CLASSIFIER_INSTRUCTIONS = """You are a strict classifier. Your ONLY job is to determine the user's intent.

Given any user input, you MUST respond with EXACTLY one of these two options - nothing else:
- SWITCH TO ETHEREUM
- SWITCH TO BITCOIN

Consider synonyms, variations, and casual phrasing (e.g. "go to eth", "switch to bitcoin", "use ethereum"). 
Output ONLY the exact phrase. No explanation, no punctuation, no extra text."""

class EType(str, Enum):
    ON_CONNECT = "session.created"
    ON_UPDATE = "session.update"

    CLIENT_MSG = "conversation.item.create"
    CLIENT_REQ_RESPONSE = "response.create"

    SERVER_TOK_STREAM = "response.output_text.delta"
    SERVER_RESPONSE_DONE = "response.done"

    ERROR = "error"

class OpenAiCli:
    def __init__(self):
        MODELNAME = "gpt-realtime"
        URL = f"wss://api.openai.com/v1/realtime?model={MODELNAME}"
        HEADERS = [f"Authorization: Bearer {OPENAI_API_KEY}"]

        self.ws = websocket.WebSocketApp(
            URL,
            header=HEADERS,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        self.cur_message = ""
        self.final_message = ""
        self.response_ready = False
        self.t = None

    def parse_classifier_response(self, raw_text):
        """Extract SWITCH TO ETHEREUM or SWITCH TO BITCOIN from GPT's response."""
        if not raw_text:
            return None
        raw_upper = raw_text.strip().upper()
        if "ETHEREUM" in raw_upper or "ETH" in raw_upper:
            return "SWITCH TO ETHEREUM"
        if "BITCOIN" in raw_upper or "BTC" in raw_upper:
            return "SWITCH TO BITCOIN"
        return None

    def get_response(self):
        while not self.response_ready:
            time.sleep(0.01)
        return self.final_message

    def set_instructions(self, instructions):
        self.ws_send(self.ws, {
            "type": EType.ON_UPDATE,
            "session": {
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": instructions,
                "tools": [],
                "tool_choice": "none"
            }
        })

    def on_open(self, ws):
        logger.info("Connected to websocket")

    def ws_send(self, ws, message):
        json_data = json.dumps(message)
        self.ws.send(json_data)

    def on_message(self, ws, message):
        if not message or not isinstance(message, str):
            return
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message received: {message[:100]}")
            return
        event_type = event.get("type")

        match event_type:
            case EType.ON_CONNECT:
                self.set_instructions(CLASSIFIER_INSTRUCTIONS)

            case EType.SERVER_TOK_STREAM:
                self.cur_message += event.get("delta", "")
                print(event.get("delta", ""), end="", flush=True)
            
            case EType.SERVER_RESPONSE_DONE:
                if self.cur_message:
                    self.final_message = self.cur_message
                    self.cur_message = ""
                    self.response_ready = True

            case EType.ERROR:
                logger.error("Websocket error")


    def on_error(self, ws, er_msg):
        logger.error(f"error {er_msg}")

    def on_close(self, ws, code, reason):
        logger.info(f"closed {code} {reason}")

    def run(self):
        self.t = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.t.start()

    def make_request(self, message):
        logger.info("transmitting..")
        self.response_ready = False

        self.ws_send(self.ws,
                     {
                         "type": EType.CLIENT_MSG,
                         "item":{
                             "type": "message",
                             "role": "user",
                             "content": [{"type": "input_text", "text": message}]

                             }
                         })
        self.ws_send(self.ws, {"type": EType.CLIENT_REQ_RESPONSE})


# ============================================================================
# BLOCK 7: VOICE ASSISTANT PARENT CLASS 
# ============================================================================

class State(str, Enum):
    WAITING = "waiting"
    LISTENING = "listening"


class VoiceAssistant:
    def __init__(self):
        #make instances of all previously defined classes + speech recogniser
        self.openai = OpenAiCli()
        self.tts = GttsCli()
        self.recogniser = sr.Recognizer()
        
        #initialise state to waiting
        self.state = State.WAITING
        
        #speech recognintion input sample rate
        self.sample_rate = 16000
        
        #num seconds to record for wakeword and input
        self.record_wakeword_seconds = 2
        self.record_input_seconds = 3
        
        #declare sample width for speech recognition library
        self.sr_sample_width = 2
    
    #record audio and process while timing how long recording and normalization takes
    def record_audio(self, seconds):
        logger.info(f"[Listening] Speak now")
        t0 = time.time()
        pynq_audio.record(seconds)
        audio_16k = normalized_pcm(pynq_audio)
        t1 = time.time()
        logger.info(f"[Timing] Record: {seconds:.1f}s, PDM -> PCM process: {t1-t0 - seconds:.1f}s, Samples: {len(audio_16k)}")
        return audio_16k
    
    def post_instruction_to_flask(self, instruction):
        """POST the chosen instruction (SWITCH TO ETHEREUM/BITCOIN) to Flask server."""
        global active_asset
        payload = {"instruction": instruction}
        try:
            response = requests.post(INSTRUCTION_ENDPOINT, json=payload, timeout=5)
            response.raise_for_status()
            active_asset = "ETH" if "ETHEREUM" in instruction else "BTC"
            logger.info(f"Instruction '{instruction}' stored successfully: {response.status_code}")
            return True
        except requests.exceptions.ConnectionError:
            logger.warning("Flask server not reachable (dummy). Instruction would be: " + instruction)
            return False
        except Exception as e:
            logger.error(f"Failed to POST instruction: {e}")
            return False

    def convert_audio_to_text(self, audio):
        #send audio data to speech recognition library
        t0 = time.time()
        audio_data = sr.AudioData(audio.tobytes(), self.sample_rate, self.sr_sample_width)
        text = self.recogniser.recognize_google(audio_data)                      
        logger.info(f"User input: {text}")
        t1 = time.time()
        logger.info(f"[Timing] audio -> text: {t1-t0:.1f}s")
        
        return text
    
    def run(self):
        logger.info("Voice Assistant Started")
        
        self.openai.run()
        self.weight_listener = WeightListener()
        self.weight_listener.start()
        
        wake_words = set(["hey", "hei", "hei", "jarvis", "hrs"])

        try:
            while 1:
                if self.state == State.WAITING:
                    #record audio
                    audio = self.record_audio(self.record_wakeword_seconds)
                    
                    try:
                        #converting audio to text with speech recogniser
                        text = self.convert_audio_to_text(audio)
                        
                        #converting input text to set and comparing it with wakeword set
                        input_text_set = set(text.lower().split(" "))
                        if len(wake_words.intersection(input_text_set)) > 0:
                            self.state = State.LISTENING
                            logger.info("Wakeword detected, changing to listening state")
      
                    except sr.UnknownValueError:
                        logger.warning("Could not understand audio")

                    except Exception as e:
                        logger.error(f"Error: {e}")
                        
                elif self.state == State.LISTENING:
                    logger.info(f"Recording command ({self.record_input_seconds} seconds)...")
                    command_audio = self.record_audio(self.record_input_seconds)
                    
                    try:
                        text = self.convert_audio_to_text(command_audio)
                        
                        t0 = time.time()
                        self.openai.make_request(text)
                        raw_response = self.openai.get_response()
                        t1 = time.time()
                        
                        logger.info(f"[Timing] WS GPT response time: {t1-t0:.1f}s")

                        instruction = self.openai.parse_classifier_response(raw_response)

                        if instruction:
                            self.post_instruction_to_flask(instruction)
                            self.tts.say(f"Switched to {'Ethereum' if 'ETHEREUM' in instruction else 'Bitcoin'}.")
                        else:
                            logger.warning(f"Could not classify input. Got: {raw_response}")
                            self.tts.say("I couldn't determine whether you meant Ethereum or Bitcoin.")
                        
                    except sr.UnknownValueError:
                        logger.warning("Could not understand audio")
                        self.state = State.WAITING
                    except Exception as e:
                        logger.error(f"Error: {e}")
                    
                    self.state = State.LISTENING
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        logger.info("Voice Assistant Stopped")

# ============================================================================
# BLOCK 8: WEIGHT LISTENER THREAD
# ============================================================================
# Uses CalculationEngineV2 from calc_engine.py to compute 12 features from Binance REST data.

import sys, os
_cwd = os.getcwd()
_parent = os.path.dirname(_cwd)
_candidates = [
    os.path.join(_cwd, "Trading Indicatiors files"), os.path.join(_cwd, "Trading Indicators files"),
    _cwd, os.path.join(_parent, "Trading Indicatiors files"), os.path.join(_parent, "Trading Indicators files"), _parent
]
for _p in _candidates:
    if os.path.isfile(os.path.join(_p, "calc_engine.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break
try:
    from calc_engine import CalculationEngineV2
except ModuleNotFoundError:
    raise ModuleNotFoundError("calc_engine.CalculationEngineV2 not found. Ensure calc_engine.py is in the same folder or 'Trading Indicatiors files'. Current dir: " + _cwd)

from binance_ws import BinanceWSClient # edit for local imports???

class TestingEngine:
    def __init__(self):
        self.binance_ws = BinanceWSClient()
        self.binance_ws.run_ws()
        self.ce = CalculationEngineV2()
        self.last_last_price = 0
        self.last_prediction = 0

        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.accuracy = 0


    def use_weights(self, weights):
        trades = self.binance_ws.trades
        last_price = self.binance_ws.last_price

        now = time.time()
        trades_snapshot = list(self.binance_ws.trades)
        trades_10 = self.binance_ws.get_trades_since(now - 11.0)
        trades_30 = self.binance_ws.get_trades_since(now - 31.0)
        shortterm_trades = self.binance_ws.get_trades_since(now - 1)

        order_book = self.binance_ws.order_book
        asks = order_book.get("asks", [])
        bids = order_book.get("bids", [])

        best_ask = asks[0]   # (price, qty)
        best_bid = bids[0]   # (price, qty)
        last_price = self.binance_ws.last_price

        indicators = {}
        # --- 12 new features ---
        indicators["spread"] = self.ce.spread_bps(best_ask[0], best_bid[0])
        indicators["wmid_dev"] = self.ce.weighted_mid_deviation(best_ask, best_bid)
        indicators["imbalance"] = self.ce.book_imbalance(asks, bids, levels=10)
        indicators["slope_ratio"] = self.ce.book_slope_ratio(asks, bids, levels=5)
        indicators["depth_r"] = self.ce.depth_ratio(asks, bids)
        indicators["vol_delta"] = self.ce.volume_delta_ratio(trades_snapshot, now, 5.0)
        indicators["intensity"] = self.ce.trade_intensity_zscore(len(shortterm_trades))
        indicators["large_ratio"] = self.ce.large_trade_ratio(trades_snapshot, now, 5.0, threshold_qty=0.1)
        indicators["volatility"] = self.ce.realized_volatility(trades_10, now, 10.0)
        indicators["mom"] = self.ce.momentum(trades_10, now, 10.0)
        indicators["vwma_dev"] = self.ce.vwma_deviation(trades_30, now, 30.0, last_price)
        indicators["cum_vdelta"] = self.ce.cumulative_volume_delta(indicators["vol_delta"])
            
        
        print('='*100)
        pred = float(np.dot(weights[:-1], np.array(list(indicators.values()))) + weights[-1])
        print(f"prediction: {pred}, actual price: {self.binance_ws.last_price}")
        prediction = ""
        if self.last_prediction - pred < 0:
            prediction = "SELL"
        else:
            prediction = "BUY"

        if self.last_last_price - self.binance_ws.last_price < 0:
            actual = "SELL"
        else:
            actual = "BUY"
        print(f"predicted signal: {prediction}, actual signal: {actual}")

        if prediction == actual:
            self.correct_predictions += 1
        else:
            self.incorrect_predictions += 1
        self.accuracy = self.correct_predictions / (self.correct_predictions + self.incorrect_predictions)
        
        print(f"accuracy: {self.accuracy*100}%")
        print('='*100)
        self.last_last_price = self.binance_ws.last_price
        self.last_prediction = pred


SLEEP_TIME = 1
def main_loop():
    test_eng = TestingEngine()
    while 1:
        response = requests.get("http://13.60.162.169:5000/weights").json()
        weights = response["weights"] 
        
        test_eng.use_weights(weights)

        time.sleep(SLEEP_TIME)

def run_main_loop():
    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    
# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Initializing Voice Assistant...")

    run_main_loop() 

    assistant = VoiceAssistant()
    assistant.run()
