import json
import websocket

DEFAULT_URL = "wss://stream.binance.com:9443/stream?streams=ethusdt@trade/ethusdt@depth20@100ms"


def on_open(ws):
    print("Connected to Binance WebSocket")


def on_message(ws, message):
    try:
        data = json.loads(message)
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print(message)


def on_error(ws, error):
    print("WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    print(f"Disconnected: code={close_status_code}, message={close_msg}")


def main():
    ws = websocket.WebSocketApp(
        DEFAULT_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()


if __name__ == "__main__":
    main()
