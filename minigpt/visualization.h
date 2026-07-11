// visualization.h
#pragma once
#include "value.h"
#include "model.h"
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <iostream>
#include <unordered_map>

#ifdef HAS_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

namespace visualization {

// Generate DOT format for computational graph
inline std::string graph_to_dot(const Value::Ptr& node, 
                                const std::string& name = "graph") {
    std::string dot = "digraph " + name + " {\n";
    dot += "  rankdir=TB;\n";
    dot += "  node [shape=box, style=filled, fillcolor=lightblue];\n\n";

    // Traverse graph (simplified)
    std::vector<Value::Ptr> visited;
    std::function<void(Value::Ptr)> traverse = [&](Value::Ptr n) {
        if (std::find(visited.begin(), visited.end(), n) != visited.end()) return;
        visited.push_back(n);

        // Node label
        std::string label = "data=" + std::to_string(n->data);
        if (!n->_op.empty()) {
            label += "\\nop=" + n->_op;
        }
        dot += "  node" + std::to_string((uintptr_t)n.get()) + 
               " [label=\"" + label + "\"];\n";

        // Edges to children
        for (auto& child : n->_prev) {
            dot += "  node" + std::to_string((uintptr_t)n.get()) + 
                   " -> node" + std::to_string((uintptr_t)child.get()) + ";\n";
            traverse(child);
        }
    };

    traverse(node);
    dot += "}\n";
    return dot;
}

// Plot attention weights (saves as CSV or uses Python)
inline void plot_attention(const std::vector<std::vector<double>>& attention,
                           const std::string& filename = "attention.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    for (size_t i = 0; i < attention.size(); ++i) {
        for (size_t j = 0; j < attention[i].size(); ++j) {
            file << attention[i][j];
            if (j + 1 < attention[i].size()) file << ",";
        }
        file << "\n";
    }
    file.close();

    std::cout << "Attention saved to " << filename << std::endl;
    std::cout << "You can visualize with: python -c \"import pandas as pd; "
              << "import matplotlib.pyplot as plt; "
              << "plt.imshow(pd.read_csv('" << filename << "').values); "
              << "plt.colorbar(); plt.show()\"" << std::endl;
}

// Save embeddings for t-SNE visualization
inline void save_embeddings(const std::vector<Value::Ptr>& embeddings,
                            const std::vector<std::string>& labels,
                            const std::string& filename = "embeddings.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "label";
    for (size_t i = 0; i < embeddings[0]->data; ++i) {
        file << ",dim" << i;
    }
    file << "\n";

    for (size_t i = 0; i < embeddings.size(); ++i) {
        file << (i < labels.size() ? labels[i] : std::to_string(i));
        // This is simplified - actual embedding access depends on your implementation
        file << "\n";
    }
    file.close();
}

// Progress bar (console)
class ConsoleProgressBar {
public:
    ConsoleProgressBar(int total, const std::string& desc = "Progress", int width = 50)
        : total(total),           // 1. total
          current(0),             // 2. current
          width(width),           // 3. width
          desc(desc) {            // 4. desc
    }

    void update(int value) {
        current = value;
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

        std::cout << "\r" << desc << ": [";
        for (int i = 0; i < width; ++i) {
            if (i < bar_width) std::cout << "=";
            else if (i == bar_width) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << static_cast<int>(progress * 100.0) << "%";
        std::cout.flush();
    }

    int total;
    int current;
    int width;
    std::string desc;
};

// Training monitor with plotting
class TrainingMonitor {
public:
    TrainingMonitor() : epoch(0) {}

    void add_metric(const std::string& name, double value) {
        metrics[name].push_back(value);
        if (epoch > 0) {
            epochs.push_back(epoch);
        }
    }

    void new_epoch() {
        epoch++;
    }

    void save_to_csv(const std::string& filename = "training_metrics.csv") {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        // Header
        file << "epoch";
        for (const auto& pair : metrics) {
            file << "," << pair.first;
        }
        file << "\n";

        // Data
        size_t max_len = 0;
        for (const auto& pair : metrics) {
            max_len = std::max(max_len, pair.second.size());
        }

        for (size_t i = 0; i < max_len; ++i) {
            file << i;
            for (const auto& pair : metrics) {
                if (i < pair.second.size()) {
                    file << "," << pair.second[i];
                } else {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    }

    void print_summary() {
        std::cout << "=== Training Summary ===" << std::endl;
        for (const auto& pair : metrics) {
            if (!pair.second.empty()) {
                std::cout << pair.first << " (last): " << pair.second.back() << std::endl;
            }
        }
    }

private:
    int epoch;
    std::unordered_map<std::string, std::vector<double>> metrics;
    std::vector<int> epochs;
};

} // namespace visualization