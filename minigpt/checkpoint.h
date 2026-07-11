// checkpoint.h
#pragma once
#include "model.h"
#include "optim.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <sstream>  // TAMBAHKAN INI!

struct CheckpointData {
    int epoch;
    double loss;
    std::vector<double> params_data;
    std::vector<double> m;  // Adam momentum
    std::vector<double> v;  // Adam variance
    int t;  // Adam step
    double lr;
};

class CheckpointManager {
public:
    CheckpointManager(const std::string& save_dir = "checkpoints")
        : save_dir(save_dir), auto_save_interval(0), max_keep(5) {
        std::filesystem::create_directories(save_dir);
    }

    // Save checkpoint
    bool save(MiniGPT& model, AdamW& optimizer, int epoch, double loss, 
              const std::string& name = "") {
        CheckpointData data;
        data.epoch = epoch;
        data.loss = loss;
        data.t = optimizer.get_t();
        data.lr = optimizer.lr;

        // Save model parameters
        auto params = model.parameters();
        data.params_data.reserve(params.size());
        for (auto& p : params) {
            data.params_data.push_back(p->data);
        }

        // Save optimizer state
        data.m = optimizer.get_m();
        data.v = optimizer.get_v();

        // Generate filename
        std::string filename = name.empty() ? 
            generate_filename(epoch, loss) : name;
        std::string path = save_dir + "/" + filename + ".ckpt";

        // Write to file
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open checkpoint file: " << path << std::endl;
            return false;
        }

        // Write metadata
        size_t num_params = data.params_data.size();
        file.write(reinterpret_cast<const char*>(&data.epoch), sizeof(data.epoch));
        file.write(reinterpret_cast<const char*>(&data.loss), sizeof(data.loss));
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        file.write(reinterpret_cast<const char*>(&data.t), sizeof(data.t));
        file.write(reinterpret_cast<const char*>(&data.lr), sizeof(data.lr));

        // Write parameter data
        file.write(reinterpret_cast<const char*>(data.params_data.data()), 
                   num_params * sizeof(double));

        // Write optimizer state
        size_t m_size = data.m.size();
        size_t v_size = data.v.size();
        file.write(reinterpret_cast<const char*>(&m_size), sizeof(m_size));
        file.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
        if (m_size > 0) {
            file.write(reinterpret_cast<const char*>(data.m.data()), 
                       m_size * sizeof(double));
            file.write(reinterpret_cast<const char*>(data.v.data()), 
                       v_size * sizeof(double));
        }

        file.close();

        // Manage old checkpoints
        if (max_keep > 0) {
            clean_old_checkpoints();
        }

        std::cout << "Checkpoint saved: " << path << std::endl;
        return true;
    }

    // Load checkpoint
    bool load(MiniGPT& model, AdamW& optimizer, int& epoch, double& loss,
              const std::string& name = "") {
        std::string path;
        if (name.empty()) {
            path = get_latest_checkpoint();
            if (path.empty()) {
                std::cerr << "No checkpoint found" << std::endl;
                return false;
            }
        } else {
            path = save_dir + "/" + name + ".ckpt";
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open checkpoint: " << path << std::endl;
            return false;
        }

        CheckpointData data;

        // Read metadata
        file.read(reinterpret_cast<char*>(&data.epoch), sizeof(data.epoch));
        file.read(reinterpret_cast<char*>(&data.loss), sizeof(data.loss));
        size_t num_params;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        file.read(reinterpret_cast<char*>(&data.t), sizeof(data.t));
        file.read(reinterpret_cast<char*>(&data.lr), sizeof(data.lr));

        // Read parameter data
        data.params_data.resize(num_params);
        file.read(reinterpret_cast<char*>(data.params_data.data()), 
                  num_params * sizeof(double));

        // Read optimizer state
        size_t m_size, v_size;
        file.read(reinterpret_cast<char*>(&m_size), sizeof(m_size));
        file.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
        if (m_size > 0) {
            data.m.resize(m_size);
            data.v.resize(v_size);
            file.read(reinterpret_cast<char*>(data.m.data()), m_size * sizeof(double));
            file.read(reinterpret_cast<char*>(data.v.data()), v_size * sizeof(double));
        }

        file.close();

        // Restore model parameters
        auto params = model.parameters();
        if (params.size() != num_params) {
            std::cerr << "Parameter count mismatch: " << params.size() 
                      << " vs " << num_params << std::endl;
            return false;
        }
        for (size_t i = 0; i < params.size(); ++i) {
            params[i]->data = data.params_data[i];
        }

        // Restore optimizer state
        if (!data.m.empty() && data.m.size() == data.v.size()) {
            optimizer.set_m(data.m);
            optimizer.set_v(data.v);
            optimizer.set_t(data.t);
            optimizer.lr = data.lr;
        }

        epoch = data.epoch;
        loss = data.loss;

        std::cout << "Checkpoint loaded: " << path << " (epoch " << epoch << ", loss " << loss << ")" << std::endl;
        return true;
    }

    // Set auto-save interval
    void set_auto_save(int interval) {
        auto_save_interval = interval;
    }

    // Set max checkpoints to keep
    void set_max_keep(int max_keep) {
        this->max_keep = max_keep;
    }

    // Get latest checkpoint path
    std::string get_latest_checkpoint() const {
        std::string latest;
        std::filesystem::file_time_type latest_time;
        bool found = false;

        for (const auto& entry : std::filesystem::directory_iterator(save_dir)) {
            if (entry.path().extension() == ".ckpt") {
                if (!found || entry.last_write_time() > latest_time) {
                    latest_time = entry.last_write_time();
                    latest = entry.path().string();
                    found = true;
                }
            }
        }

        return latest;
    }

    // List all checkpoints
    std::vector<std::string> list_checkpoints() const {
        std::vector<std::string> checkpoints;
        for (const auto& entry : std::filesystem::directory_iterator(save_dir)) {
            if (entry.path().extension() == ".ckpt") {
                checkpoints.push_back(entry.path().stem().string());
            }
        }
        std::sort(checkpoints.begin(), checkpoints.end());
        return checkpoints;
    }

    // Delete old checkpoints
    void clean_old_checkpoints() {
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(save_dir)) {
            if (entry.path().extension() == ".ckpt") {
                files.push_back(entry.path());
            }
        }

        // Sort by modification time (newest first)
        std::sort(files.begin(), files.end(),
                  [](const auto& a, const auto& b) {
                      return std::filesystem::last_write_time(a) >
                             std::filesystem::last_write_time(b);
                  });

        // Keep only max_keep newest
        while (files.size() > static_cast<size_t>(max_keep)) {
            std::filesystem::remove(files.back());
            files.pop_back();
        }
    }

private:
    std::string generate_filename(int epoch, double loss) const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        ss << "_epoch" << epoch << "_loss" << std::fixed << std::setprecision(4) << loss;
        return ss.str();
    }

    std::string save_dir;
    int auto_save_interval;
    int max_keep;
};