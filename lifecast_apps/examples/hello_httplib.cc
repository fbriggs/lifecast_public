// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
Example usage:
bazel run //examples:hello_httplib -- --port 8123
*/
#include "gflags/gflags.h"
#include "source/logger.h"
#include <thread>
#include <chrono>

#define _WIN32_WINNT 0x0A00 // Targeting Windows 10
#include "third_party/httplib.h"

DEFINE_int32(port, 8080, "Port to run the HTTP server.");
DEFINE_int32(srv_time_limit, 0, "Time limit in seconds for the server to run. Default: no limit");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    httplib::Server svr;
    XPLINFO << "Starting HTTP Server on port " << FLAGS_port;

    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        XPLINFO << "Received GET request from " << req.remote_addr;
        res.set_content("Hello World", "text/plain");
    });

    svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
        XPLINFO << "Received POST request from " << req.remote_addr;
        if (!req.body.empty()) {
            XPLINFO << "Request body: " << req.body;
        }
        res.set_content("Hello World", "text/plain");
    });

    // Start the server in a separate thread
    std::thread server_thread([&svr]() {
        svr.listen("0.0.0.0", FLAGS_port);
    });

    // Demonstration of stopping the server
    if (FLAGS_srv_time_limit > 0) {
        XPLINFO << "Server will stop after " << FLAGS_srv_time_limit << " seconds.";
        std::this_thread::sleep_for(std::chrono::seconds(FLAGS_srv_time_limit));
        svr.stop();
    }

    // Join the server thread to wait for its termination
    server_thread.join();

    XPLINFO << "HTTP Server has stopped.";
    return 0;
}
