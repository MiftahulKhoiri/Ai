// config.h
#pragma once
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>  // Butuh library json, install: pip install nlohmann-json

using json = nlohmann::json;

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
    std::string activation = "gelu";  // "gelu", "relu", "swish"
    bool use_bias = true;
    double init_std = 0.02;
    bool share_embeddings = false;
    std::string device = "cpu";  // "cpu", "cuda"
    
    // Load from JSON
    static ModelConfig from_json(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + path);
        }
        json j;
        file >> j;
        return from_json(j);
    }
    
    static ModelConfig from_json(const json& j) {
        ModelConfig config;
        if (j.contains("vocab_size")) config.vocab_size = j["vocab_size"];
        if (j.contains("d_model")) config.d_model = j["d_model"];
        if (j.contains("n_heads")) config.n_heads = j["n_heads"];
        if (j.contains("n_layers")) config.n_layers = j["n_layers"];
        if (j.contains("d_ff")) config.d_ff = j["d_ff"];
        if (j.contains("max_len")) config.max_len = j["max_len"];
        if (j.contains("dropout")) config.dropout = j["dropout"];
        if (j.contains("layer_norm_eps")) config.layer_norm_eps = j["layer_norm_eps"];
        if (j.contains("learning_rate")) config.learning_rate = j["learning_rate"];
        if (j.contains("weight_decay")) config.weight_decay = j["weight_decay"];
        if (j.contains("beta1")) config.beta1 = j["beta1"];
        if (j.contains("beta2")) config.beta2 = j["beta2"];
        if (j.contains("eps")) config.eps = j["eps"];
        if (j.contains("warmup_steps")) config.warmup_steps = j["warmup_steps"];
        if (j.contains("total_steps")) config.total_steps = j["total_steps"];
        if (j.contains("batch_size")) config.batch_size = j["batch_size"];
        if (j.contains("epochs")) config.epochs = j["epochs"];
        if (j.contains("temperature")) config.temperature = j["temperature"];
        if (j.contains("top_k")) config.top_k = j["top_k"];
        if (j.contains("top_p")) config.top_p = j["top_p"];
        if (j.contains("use_cache")) config.use_cache = j["use_cache"];
        if (j.contains("activation")) config.activation = j["activation"];
        if (j.contains("use_bias")) config.use_bias = j["use_bias"];
        if (j.contains("init_std")) config.init_std = j["init_std"];
        if (j.contains("share_embeddings")) config.share_embeddings = j["share_embeddings"];
        if (j.contains("device")) config.device = j["device"];
        return config;
    }
    
    // Save to JSON
    void to_json(const std::string& path) const {
        json j;
        j["vocab_size"] = vocab_size;
        j["d_model"] = d_model;
        j["n_heads"] = n_heads;
        j["n_layers"] = n_layers;
        j["d_ff"] = d_ff;
        j["max_len"] = max_len;
        j["dropout"] = dropout;
        j["layer_norm_eps"] = layer_norm_eps;
        j["learning_rate"] = learning_rate;
        j["weight_decay"] = weight_decay;
        j["beta1"] = beta1;
        j["beta2"] = beta2;
        j["eps"] = eps;
        j["warmup_steps"] = warmup_steps;
        j["total_steps"] = total_steps;
        j["batch_size"] = batch_size;
        j["epochs"] = epochs;
        j["temperature"] = temperature;
        j["top_k"] = top_k;
        j["top_p"] = top_p;
        j["use_cache"] = use_cache;
        j["activation"] = activation;
        j["use_bias"] = use_bias;
        j["init_std"] = init_std;
        j["share_embeddings"] = share_embeddings;
        j["device"] = device;
        
        std::ofstream file(path);
        file << j.dump(4);
        file.close();
    }
};