// server.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include "generation_advanced.h"
#include <string>
#include <map>
#include <functional>
#include <thread>
#include <iostream>

class HTTPServer {
public:
    HTTPServer(int port = 8080) : port(port), running(false) {}
    virtual ~HTTPServer() { stop(); }
    
    void add_route(const std::string& path, const std::string& method, 
                   std::function<std::string(const std::map<std::string, std::string>&)> handler) {
        routes[path + ":" + method] = handler;
    }
    
    void start() {
        running = true;
        server_thread = std::thread(&HTTPServer::run, this);
        std::cout << "🌐 Server started on port " << port << std::endl;
    }
    
    void stop() {
        running = false;
        if (server_thread.joinable()) {
            server_thread.join();
        }
    }
    
private:
    void run() {
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    int port;
    bool running;
    std::thread server_thread;
    std::map<std::string, std::function<std::string(const std::map<std::string, std::string>&)>> routes;
};

class GPTAPIServer {
public:
    GPTAPIServer(MiniGPT& model, ByteLevelBPETokenizer& tokenizer, int port = 8080)
        : m_model(model), 
          m_tokenizer(tokenizer), 
          server(port), 
          generator(model, tokenizer) {
        // Suppress unused parameter warnings
        (void)m_model;
        (void)m_tokenizer;
        setup_routes();
    }
    
    void start() { server.start(); }
    void stop() { server.stop(); }
    
private:
    void setup_routes() {
        // Generate endpoint
        server.add_route("/generate", "POST", [this](const std::map<std::string, std::string>& params) {
            auto prompt_it = params.find("prompt");
            if (prompt_it == params.end()) {
                return std::string("{\"error\":\"prompt required\"}");
            }
            
            int max_tokens = 50;
            auto max_it = params.find("max_tokens");
            if (max_it != params.end()) {
                max_tokens = std::stoi(max_it->second);
            }
            
            // Configure generation
            advanced_generation::GenerationConfig config;
            config.max_length = max_tokens;
            config.temperature = 0.8;
            config.top_p = 0.9;
            config.top_k = 40;
            generator.set_config(config);
            
            auto results = generator.generate(prompt_it->second);
            
            // Return JSON response
            std::string response = "{\"generated\":[";
            for (size_t i = 0; i < results.size(); ++i) {
                response += "\"" + escape_json(results[i]) + "\"";
                if (i + 1 < results.size()) response += ",";
            }
            response += "]}";
            return response;
        });
        
        // Status endpoint
        server.add_route("/status", "GET", [](const std::map<std::string, std::string>&) {
            return std::string("{\"status\":\"running\",\"model\":\"MiniGPT\"}");
        });
        
        // Health check
        server.add_route("/health", "GET", [](const std::map<std::string, std::string>&) {
            return std::string("{\"status\":\"healthy\"}");
        });
    }
    
    std::string escape_json(const std::string& str) {
        std::string escaped;
        for (char c : str) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b"; break;
                case '\f': escaped += "\\f"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }
        return escaped;
    }
    
    // Ganti nama jadi m_ prefix (opsional, untuk menunjukkan member)
    MiniGPT& m_model;
    ByteLevelBPETokenizer& m_tokenizer;
    HTTPServer server;
    advanced_generation::AdvancedGenerator generator;
};