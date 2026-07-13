// server.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include "generation_advanced.h"
#include "httplib.h"   // https://github.com/yhirose/cpp-httplib -- taruh httplib.h sejajar file ini
#include <string>
#include <iostream>
#include <stdexcept>

// Server HTTP asli (pakai cpp-httplib) untuk expose MiniGPT lewat port
// web. cpp-httplib multi-thread secara default -- tiap request ditangani
// di thread pool internalnya sendiri.
//
// DESAIN THREAD-SAFETY:
// - Bobot model (MiniGPT::parameters()) hanya DIBACA saat inference,
//   tidak pernah ditulis selama server jalan -> aman dibaca paralel.
// - Tokenizer (encode/decode) hanya membaca vocab/merge_rank, tidak
//   pernah ditulis setelah training selesai -> aman dipanggil paralel.
// - KV-cache TIDAK lagi disimpan di MiniGPT (state global) -- tiap
//   request membuat AdvancedGenerator baru, yang otomatis membuat cache
//   miliknya sendiri (lihat generation_advanced.h/.cpp). Jadi request
//   paralel tidak saling menimpa cache satu sama lain.
// - model.set_training(false) dipanggil SEKALI di constructor server
//   ini (bukan per-request), supaya tidak ada race pada flag dropout.
class GPTAPIServer {
public:
    GPTAPIServer(MiniGPT& model, ByteLevelBPETokenizer& tokenizer, int port = 8080)
        : m_model(model),
          m_tokenizer(tokenizer),
          port(port) {
        // Set mode eval SEKALI di sini, bukan per-request.
        m_model.set_training(false);
        setup_routes();
    }

    // Blocking -- panggil ini dari main thread (atau thread terpisah
    // kalau ingin non-blocking dari sisi pemanggil).
    void start() {
        std::cout << "🌐 Server mendengarkan di port " << port << std::endl;
        svr.listen("0.0.0.0", port);
    }

    void stop() {
        svr.stop();
    }

private:
    void setup_routes() {
        svr.Post("/generate", [this](const httplib::Request& req, httplib::Response& res) {
            handle_generate(req, res);
        });

        svr.Get("/status", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"running\",\"model\":\"MiniGPT\"}", "application/json");
        });

        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"healthy\"}", "application/json");
        });
    }

    void handle_generate(const httplib::Request& req, httplib::Response& res) {
        std::string prompt;
        int max_tokens = 50;
        float temperature = 0.8f;
        float top_p = 0.9f;
        int top_k = 40;

        try {
            if (!parse_json_string(req.body, "prompt", prompt) || prompt.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"prompt required\"}", "application/json");
                return;
            }
            parse_json_int(req.body, "max_tokens", max_tokens);
            parse_json_float(req.body, "temperature", temperature);
            parse_json_float(req.body, "top_p", top_p);
            parse_json_int(req.body, "top_k", top_k);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON body\"}", "application/json");
            return;
        }

        // FIX (server paralel): AdvancedGenerator baru per request --
        // ini yang membuat konfigurasi (config member) DAN cache KV
        // tidak lagi di-share antar request yang jalan bersamaan.
        // Konstruksinya ringan (cuma reference + struct config kecil),
        // jadi aman dibuat berulang tiap request.
        advanced_generation::AdvancedGenerator generator(m_model, m_tokenizer);

        advanced_generation::GenerationConfig gen_config;
        gen_config.max_length = max_tokens;
        gen_config.temperature = temperature;
        gen_config.top_p = top_p;
        gen_config.top_k = top_k;
        generator.set_config(gen_config);

        std::vector<std::string> results;
        try {
            results = generator.generate(prompt);
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":\"generation failed\"}", "application/json");
            return;
        }

        std::string response = "{\"generated\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            response += "\"" + escape_json(results[i]) + "\"";
            if (i + 1 < results.size()) response += ",";
        }
        response += "]}";
        res.set_content(response, "application/json");
    }

    // Parser JSON minimal (khusus field top-level yang dibutuhkan
    // endpoint ini) -- pola yang sama dipakai di config.h/tokenizer.cpp
    // untuk menghindari dependency library JSON eksternal.
    static bool parse_json_string(const std::string& json, const std::string& key, std::string& out) {
        std::string search_key = "\"" + key + "\"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return false;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        if (pos >= json.size() || json[pos] != '"') return false;
        pos++;
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    default: result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }
        out = result;
        return true;
    }

    static bool parse_json_int(const std::string& json, const std::string& key, int& out) {
        std::string search_key = "\"" + key + "\"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return false;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        size_t start = pos;
        while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos])) || json[pos] == '-')) pos++;
        if (pos == start) return false;
        try { out = std::stoi(json.substr(start, pos - start)); } catch (...) { return false; }
        return true;
    }

    static bool parse_json_float(const std::string& json, const std::string& key, float& out) {
        std::string search_key = "\"" + key + "\"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return false;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        size_t start = pos;
        while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos])) || json[pos] == '-' || json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E')) pos++;
        if (pos == start) return false;
        try { out = std::stof(json.substr(start, pos - start)); } catch (...) { return false; }
        return true;
    }

    static std::string escape_json(const std::string& str) {
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

    MiniGPT& m_model;
    ByteLevelBPETokenizer& m_tokenizer;
    int port;
    httplib::Server svr;
};