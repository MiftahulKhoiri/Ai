// metrics.h
#pragma once
#include "value.h"
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_map>  // TAMBAHKAN INI!
#include <iostream>
#include <algorithm>

namespace metrics {
    // Perplexity: exp(loss)
    inline double perplexity(double loss) {
        return std::exp(loss);
    }

    // Accuracy for classification
    inline double accuracy(const std::vector<int>& predictions, const std::vector<int>& targets) {
        if (predictions.empty() || predictions.size() != targets.size()) return 0.0;
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == targets[i]) correct++;
        }
        return static_cast<double>(correct) / predictions.size();
    }

    // BLEU score (simplified for words)
    inline double bleu_score(const std::string& reference, const std::string& candidate) {
        // Simple implementation: count common words
        std::vector<std::string> ref_words, cand_words;
        // Split by spaces (simple tokenization)
        std::string word;
        for (char c : reference) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!word.empty()) {
                    ref_words.push_back(word);
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        if (!word.empty()) ref_words.push_back(word);

        word.clear();
        for (char c : candidate) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!word.empty()) {
                    cand_words.push_back(word);
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        if (!word.empty()) cand_words.push_back(word);

        if (ref_words.empty() || cand_words.empty()) return 0.0;

        // Count matches
        int matches = 0;
        for (const auto& rw : ref_words) {
            for (const auto& cw : cand_words) {
                if (rw == cw) {
                    matches++;
                    break;
                }
            }
        }

        // Precision + brevity penalty
        double precision = static_cast<double>(matches) / cand_words.size();
        double brevity = 1.0;
        if (ref_words.size() > cand_words.size()) {
            brevity = std::exp(1.0 - static_cast<double>(ref_words.size()) / cand_words.size());
        }

        return precision * brevity;
    }

    // Training logger
    class TrainingLogger {
    public:
        TrainingLogger(const std::string& name = "training")
            : name(name), start_time(std::chrono::steady_clock::now()) {}

        void log_epoch(int epoch, double loss, double accuracy = 0.0, double lr = 0.0) {
            epochs.push_back(epoch);
            losses.push_back(loss);
            accuracies.push_back(accuracy);
            lrs.push_back(lr);
        }

        void log_metric(const std::string& key, double value) {
            metrics[key].push_back(value);
        }

        void save_logs(const std::string& path = "logs.csv") {
            std::ofstream file(path);
            if (!file.is_open()) return;

            file << "epoch,loss,accuracy,lr\n";
            for (size_t i = 0; i < epochs.size(); ++i) {
                file << epochs[i] << ","
                     << std::fixed << std::setprecision(6) << losses[i] << ","
                     << std::fixed << std::setprecision(4) << accuracies[i] << ","
                     << std::fixed << std::setprecision(8) << lrs[i] << "\n";
            }
            file.close();
        }

        void print_summary() {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

            std::cout << "=== Training Summary ===" << std::endl;
            std::cout << "Name: " << name << std::endl;
            std::cout << "Total epochs: " << epochs.size() << std::endl;
            std::cout << "Time: " << duration.count() / 60 << "m " << duration.count() % 60 << "s" << std::endl;

            if (!losses.empty()) {
                std::cout << "Final loss: " << losses.back() << std::endl;
                std::cout << "Best loss: " << *std::min_element(losses.begin(), losses.end()) << std::endl;
            }

            if (!accuracies.empty() && accuracies[0] > 0.0) {
                std::cout << "Final accuracy: " << accuracies.back() * 100.0 << "%" << std::endl;
                std::cout << "Best accuracy: " << *std::max_element(accuracies.begin(), accuracies.end()) * 100.0 << "%" << std::endl;
            }

            // Print custom metrics
            for (const auto& pair : metrics) {
                const auto& values = pair.second;
                if (!values.empty()) {
                    std::cout << pair.first << ": " << values.back() << std::endl;
                }
            }
        }

        const std::vector<double>& get_losses() const { return losses; }
        const std::vector<double>& get_accuracies() const { return accuracies; }
        const std::vector<double>& get_lrs() const { return lrs; }

    private:
        std::string name;
        std::vector<int> epochs;
        std::vector<double> losses, accuracies, lrs;
        std::unordered_map<std::string, std::vector<double>> metrics;
        std::chrono::steady_clock::time_point start_time;
    };

    // Progress bar
    class ProgressBar {
    public:
        ProgressBar(int total, const std::string& desc = "Progress")
            : total(total),           // 1. total
              width(50),              // 2. width
              desc(desc),             // 3. desc
              current(0) {            // 4. current
            start_time = std::chrono::steady_clock::now();
        }

        void update(int current_val) {
            current = current_val;
            print();
        }

        void increment() {
            current++;
            print();
        }

        void finish() {
            current = total;
            print();
            std::cout << std::endl;
        }

    private:
        void print() {
            float progress = static_cast<float>(current) / total;
            int bar_width = static_cast<int>(progress * width);

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

            std::cout << "\r" << desc << ": [";
            for (int i = 0; i < width; ++i) {
                if (i < bar_width) std::cout << "=";
                else if (i == bar_width) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << static_cast<int>(progress * 100.0) << "% ";
            std::cout << "[" << elapsed.count() << "s]";
            std::cout.flush();
        }

        int total;
        int width;
        std::string desc;
        int current;
        std::chrono::steady_clock::time_point start_time;
    };
}