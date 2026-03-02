cmake_minimum_required(VERSION 3.14)
project(binance_ws CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# IXWebSocket — WebSocket + TLS client
FetchContent_Declare(
    ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG        v11.4.5
)

# nlohmann/json — single-header JSON library
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

set(USE_TLS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ixwebsocket json)

add_executable(binance_ws main.cpp)
target_link_libraries(binance_ws PRIVATE ixwebsocket nlohmann_json::nlohmann_json)
