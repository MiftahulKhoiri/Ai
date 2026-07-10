// value.h
#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

namespace autograd {

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad{0.0};

    using Ptr = std::shared_ptr<Value>;

    static Ptr create(double data);
    
    // Operasi matematika
    Ptr operator+(Ptr other);
    Ptr operator-(Ptr other);
    Ptr operator*(Ptr other);
    Ptr operator/(Ptr other);
    Ptr tanh();
    Ptr relu();

    // Reset gradien node ini saja
    void zero_grad();

    // Backward pass dari node ini (biasanya loss)
    void backward();

private:
    explicit Value(double d) : data(d) {}

    struct Edge {
        std::weak_ptr<Value> parent;
        double gradient_factor;        // d(out)/d(parent)
    };

    std::vector<Edge> _prev;
    std::function<void()> _backward_fn;

    // Untuk topological sort
    void _build_topo(std::vector<Value*>& order, std::unordered_set<Value*>& visited);
    
    // Graph boleh mengakses anggota privat
    friend class Graph;
};

} // namespace autograd