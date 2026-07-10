// autograd.h
#pragma once
#include "graph.h"
#include "value.h"
#include <vector>

namespace autograd {

class AutogradEngine {
public:
    Graph graph;

    // Membuat node dan otomatis mendaftarkannya ke graph
    Value::Ptr createValue(double data) {
        auto v = Value::create(data);
        graph.add_node(v);
        return v;
    }

    // Menjalankan backward untuk setiap loss – gradien terakumulasi
    void backward(const std::vector<Value::Ptr>& losses) {
        for (auto& loss : losses) {
            loss->backward();
        }
    }

    // Mereset semua gradien di graph menjadi nol
    void zero_grad() {
        graph.zero_grad();
    }
};

} // namespace autograd