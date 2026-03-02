#include <iostream>
#include <deque>
#include <vector>
#include <mutex>
#include <string>
#include <utility>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <chrono>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static constexpr std::size_t MAX_TRADE_HISTORY = 50000;

struct Trade {
    double price;
    double volume;
    double time;
    bool   is_buyer_maker;
};

struct PriceLevel {
    double price;
    double qty;
};

class BinanceWSClient {
public:
    BinanceWSClient()
    {
        ws_.setUrl("wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms");

        ws_.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
            switch (msg->type) {
                case ix::WebSocketMessageType::Open:
                    std::cout << "WebSocket Open\n";
                    break;
                case ix::WebSocketMessageType::Close:
                    std::cout << "WebSocket Closed\n";
                    break;
                case ix::WebSocketMessageType::Error:
                    std::cerr << "Error: " << msg->errorInfo.reason << "\n";
                    break;
                case ix::WebSocketMessageType::Message:
                    handle_message(msg->str);
                    break;
                default:
                    break;
            }
        });
    }

    void run()
    {
        ws_.start();

        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            std::lock_guard<std::mutex> lock(mtx_);
            while (!trades_.empty()) {
                const Trade& t = trades_.back();
                std::cout << "price = " << t.price
                          << ", volume = " << t.volume
                          << ", timestamp = " << t.time << "\n";
                trades_.pop_back();
            }
        }
    }

private:
    void handle_message(const std::string& raw)
    {
        json message = json::parse(raw, nullptr, false);
        if (message.is_discarded()) return;

        std::string stream = message.value("stream", "");
        json data = message.value("data", json::object());

        if (stream.find("trade") != std::string::npos) {
            process_trade(data);
        }
        if (stream.find("depth") != std::string::npos) {
            process_order_book(data);
        }
    }

    void process_trade(const json& data)
    {
        Trade t;
        t.price          = std::stod(data["p"].get<std::string>());
        t.volume         = std::stod(data["q"].get<std::string>());
        t.time           = data["T"].get<double>() / 1000.0;
        t.is_buyer_maker = data["m"].get<bool>();

        std::lock_guard<std::mutex> lock(mtx_);
        trades_.push_back(t);
        if (trades_.size() > MAX_TRADE_HISTORY)
            trades_.pop_front();

        last_price_ = t.price;
        trade_count_++;
    }

    void process_order_book(const json& data)
    {
        auto parse_levels = [](const json& arr) {
            std::vector<PriceLevel> levels;
            levels.reserve(arr.size());
            for (auto& entry : arr) {
                double price = std::stod(entry[0].get<std::string>());
                double qty   = std::stod(entry[1].get<std::string>());
                levels.push_back({price, qty});
            }
            return levels;
        };

        std::vector<PriceLevel> asks = parse_levels(data.value("asks", json::array()));
        std::vector<PriceLevel> bids = parse_levels(data.value("bids", json::array()));

        {
            std::lock_guard<std::mutex> lock(mtx_);
            asks_ = std::move(asks);
            bids_ = std::move(bids);
            depth_count_++;
        }

        if (!asks_.empty() && !bids_.empty()) {
            double qty_diff = asks_[0].qty - bids_[0].qty;
            std::cout << "asks: (" << asks_[0].price << ", " << asks_[0].qty << ")"
                      << ", bids: (" << bids_[0].price << ", " << bids_[0].qty << ")"
                      << ", qty diff: " << qty_diff << "\n";
        }
    }

    ix::WebSocket ws_;
    std::mutex    mtx_;

    std::deque<Trade>        trades_;
    std::vector<PriceLevel>  asks_;
    std::vector<PriceLevel>  bids_;

    double                   last_price_  = 0.0;
    std::size_t              depth_count_ = 0;
    std::size_t              trade_count_ = 0;
};

int main()
{
    ix::initNetSystem();

    BinanceWSClient client;
    client.run();

    ix::uninitNetSystem();
    return 0;
}
