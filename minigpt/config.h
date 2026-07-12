// config.h
#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <cctype>
#include <unordered_map>
#include <iostream>
#include <type_traits>   // FIX: dipakai untuk std::is_same_v di parse_value

struct ModelConfig {
    // Architecture
    int vocab_size = 50257;
    int d_model = 768;
    int n_heads = 12;
    int n_layers = 12;
    int d_ff = 3072;
    int max_len = 1024;
    double dropout = 0.1;
    double layer_norm_eps = 1e-5;

    // Training
    double learning_rate = 1e-3;
    double weight_decay = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    int warmup_steps = 1000;
    int total_steps = 100000;
    int batch_size = 8;
    int epochs = 10;

    // Generation
    double temperature = 0.8;
    int top_k = 40;
    double top_p = 0.9;
    bool use_cache = true;

    // Other
    std::string activation = "gelu";
    bool use_bias = true;
    double init_std = 0.02;
    bool share_embeddings = false;
    std::string device = "cpu";

    // ============ JSON Parser Tanpa Library Eksternal ============

    // Load from JSON file
    static ModelConfig from_json(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "⚠️  Warning: Cannot open config file: " << path << ", using defaults" << std::endl;
            return ModelConfig();
        }

        std::string content;
        file.seekg(0, std::ios::end);
        content.reserve(file.tellg());
        file.seekg(0, std::ios::beg);
        content.assign((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
        file.close();

        ModelConfig config;

        parse_value(content, "vocab_size", config.vocab_size);
        parse_value(content, "d_model", config.d_model);
        parse_value(content, "n_heads", config.n_heads);
        parse_value(content, "n_layers", config.n_layers);
        parse_value(content, "d_ff", config.d_ff);
        parse_value(content, "max_len", config.max_len);
        parse_value(content, "dropout", config.dropout);
        parse_value(content, "layer_norm_eps", config.layer_norm_eps);
        parse_value(content, "learning_rate", config.learning_rate);
        parse_value(content, "weight_decay", config.weight_decay);
        parse_value(content, "beta1", config.beta1);
        parse_value(content, "beta2", config.beta2);
        parse_value(content, "eps", config.eps);
        parse_value(content, "warmup_steps", config.warmup_steps);
        parse_value(content, "total_steps", config.total_steps);
        parse_value(content, "batch_size", config.batch_size);
        parse_value(content, "epochs", config.epochs);
        parse_value(content, "temperature", config.temperature);
        parse_value(content, "top_k", config.top_k);
        parse_value(content, "top_p", config.top_p);
        parse_value(content, "use_cache", config.use_cache);
        parse_value(content, "activation", config.activation);
        parse_value(content, "use_bias", config.use_bias);
        parse_value(content, "init_std", config.init_std);
        parse_value(content, "share_embeddings", config.share_embeddings);
        parse_value(content, "device", config.device);

        return config;
    }

    // Save to JSON file
    void to_json(const std::string& path) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "⚠️  Warning: Cannot save config file: " << path << std::endl;
            return;
        }

        file << "{\n";
        file << "    \"vocab_size\": " << vocab_size << ",\n";
        file << "    \"d_model\": " << d_model << ",\n";
        file << "    \"n_heads\": " << n_heads << ",\n";
        file << "    \"n_layers\": " << n_layers << ",\n";
        file << "    \"d_ff\": " << d_ff << ",\n";
        file << "    \"max_len\": " << max_len << ",\n";
        file << "    \"dropout\": " << dropout << ",\n";
        file << "    \"layer_norm_eps\": " << layer_norm_eps << ",\n";
        file << "    \"learning_rate\": " << learning_rate << ",\n";
        file << "    \"weight_decay\": " << weight_decay << ",\n";
        file << "    \"beta1\": " << beta1 << ",\n";
        file << "    \"beta2\": " << beta2 << ",\n";
        file << "    \"eps\": " << eps << ",\n";
        file << "    \"warmup_steps\": " << warmup_steps << ",\n";
        file << "    \"total_steps\": " << total_steps << ",\n";
        file << "    \"batch_size\": " << batch_size << ",\n";
        file << "    \"epochs\": " << epochs << ",\n";
        file << "    \"temperature\": " << temperature << ",\n";
        file << "    \"top_k\": " << top_k << ",\n";
        file << "    \"top_p\": " << top_p << ",\n";
        file << "    \"use_cache\": " << (use_cache ? "true" : "false") << ",\n";
        // FIX: escape string sebelum ditulis ke JSON, supaya kalau nanti
        // nilainya diganti user ke sesuatu yang berisi '"' atau '\', file
        // JSON yang dihasilkan tetap valid dan bisa di-parse balik.
        file << "    \"activation\": \"" << escape_json(activation) << "\",\n";
        file << "    \"use_bias\": " << (use_bias ? "true" : "false") << ",\n";
        file << "    \"init_std\": " << init_std << ",\n";
        file << "    \"share_embeddings\": " << (share_embeddings ? "true" : "false") << ",\n";
        file << "    \"device\": \"" << escape_json(device) << "\"\n";
        file << "}\n";
        file.close();
    }

private:
    // Helper: escape string untuk output JSON yang valid
    static std::string escape_json(const std::string& str) {
        std::string escaped;
        escaped.reserve(str.size());
        for (char c : str) {
            switch (c) {
                case '"':  escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n";  break;
                case '\t': escaped += "\\t";  break;
                case '\r': escaped += "\\r";  break;
                default:   escaped += c;      break;
            }
        }
        return escaped;
    }

    // Helper: trim whitespace
    static void trim(std::string& str) {
        // FIX: cast eksplisit ke unsigned char sebelum dipanggil ke
        // std::isspace. char di banyak platform (termasuk ARM/Android)
        // adalah signed -- kalau ada byte > 127 (mis. sisa karakter
        // non-ASCII), nilainya jadi negatif dan meneruskannya ke fungsi
        // <cctype> adalah undefined behavior menurut standar C.
        str.erase(0, str.find_first_not_of(" \t\n\r"));
        str.erase(str.find_last_not_of(" \t\n\r") + 1);
    }

    // Helper: find value by key in JSON string
    static std::string find_value(const std::string& json, const std::string& key) {
        std::string search_key = "\"" + key + "\"";
        size_t pos = json.find(search_key);
        if (pos == std::string::npos) return "";

        pos = json.find(':', pos);
        if (pos == std::string::npos) return "";
        pos++;

        // FIX: cast ke unsigned char -- lihat penjelasan di trim()
        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
        if (pos >= json.size()) return "";

        size_t end_pos = pos;

        if (json[pos] == '"') {
            pos++;
            end_pos = json.find('"', pos);
            if (end_pos == std::string::npos) return "";
            return json.substr(pos, end_pos - pos);
        } else if (json[pos] == 't' || json[pos] == 'f') {
            // FIX: cast ke unsigned char
            while (pos < json.size() && std::isalpha(static_cast<unsigned char>(json[pos]))) pos++;
            return json.substr(end_pos, pos - end_pos);
        } else {
            // FIX: cast ke unsigned char
            while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos])) || json[pos] == '.' || json[pos] == '-' || json[pos] == 'e' || json[pos] == 'E')) {
                pos++;
            }
            return json.substr(end_pos, pos - end_pos);
        }
    }

    // Template untuk parsing nilai
    template<typename T>
    static void parse_value(const std::string& json, const std::string& key, T& value) {
        std::string val = find_value(json, key);
        if (val.empty()) return;
        trim(val);

        if constexpr (std::is_same_v<T, int>) {
            try { value = std::stoi(val); } catch (...) {}
        } else if constexpr (std::is_same_v<T, double>) {
            try { value = std::stod(val); } catch (...) {}
        } else if constexpr (std::is_same_v<T, bool>) {
            if (val == "true" || val == "True" || val == "1") value = true;
            else if (val == "false" || val == "False" || val == "0") value = false;
        } else if constexpr (std::is_same_v<T, std::string>) {
            value = val;
        }
    }
};

// ============ Untuk Binding Python ============
inline ModelConfig load_config(const std::string& path) {
    return ModelConfig::from_json(path);
}

inline void save_config(const ModelConfig& config, const std::string& path) {
    config.to_json(path);
}