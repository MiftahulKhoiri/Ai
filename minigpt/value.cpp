// value.cpp
#include "value.h"
#include "backward.h"
#include <cmath>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <utility>

Value::Value(double data, std::vector<Value::Ptr> children, std::string op)
    : data(data), grad(0.0), _prev(children), _op(op) {}

Value::Ptr Value::create(double data) {
    return std::make_shared<Value>(data);
}

void Value::backward() {
    std::vector<Value::Ptr> topo;
    std::unordered_set<Value*> visited;

    std::vector<std::pair<Value::Ptr, size_t>> stack;
    stack.push_back({shared_from_this(), 0});
    visited.insert(this);

    while (!stack.empty()) {
        auto& [node, idx] = stack.back();

        if (idx < node->_prev.size()) {
            Value::Ptr child = node->_prev[idx];
            idx++;
            if (visited.find(child.get()) == visited.end()) {
                visited.insert(child.get());
                stack.push_back({child, 0});
            }
        } else {
            topo.push_back(node);
            stack.pop_back();
        }
    }

    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) {
            (*it)->_backward();
        }
    }
}

std::string Value::repr() const {
    std::stringstream ss;
    ss << "Value(data=" << data << ", grad=" << grad;
    if (!_op.empty()) {
        ss << ", op=" << _op;
    }
    ss << ")";
    return ss.str();
}

// ============================================================
// Operator implementations
//
// FIX PENTING: setiap "out->_backward = [out, ...]" SEBELUMNYA meng-
// capture "out" (shared_ptr ke dirinya sendiri) di dalam closure yang
// disimpan sebagai member out->_backward. Ini reference cycle: out ->
// _backward -> capture out -> (kembali ke out). Akibatnya refcount out
// TIDAK PERNAH turun ke 0 walau tidak ada lagi yang menunjuk ke situ
// dari luar -> setiap Value node yang pernah dibuat lewat operator
// manapun BOCOR PERMANEN. Karena forward+backward pass membuat ribuan
// node baru, RAM naik terus setiap batch tanpa pernah turun.
//
// FIX: precompute raw pointer (out.get()) SEBELUM membuat closure, lalu
// capture raw pointer itu (bukan shared_ptr "out" itu sendiri). Operand
// lain (a, b) tetap aman di-capture sebagai shared_ptr karena mereka
// tidak balik menunjuk ke "out" (a dan b adalah node upstream/sebelum
// out di graph, bukan downstream).
// ============================================================

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data + b->data, 
                                        std::vector<ValuePtr>{a, b}, 
                                        "+");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a, b]() {
        autograd::backward::add(out_ptr, a, b);
    };
    return out;
}

ValuePtr operator+(const ValuePtr& a, double b) {
    return a + Value::create(b);
}

ValuePtr operator+(double a, const ValuePtr& b) {
    return Value::create(a) + b;
}

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data - b->data, 
                                        std::vector<ValuePtr>{a, b}, 
                                        "-");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a, b]() {
        autograd::backward::sub(out_ptr, a, b);
    };
    return out;
}

ValuePtr operator-(const ValuePtr& a, double b) {
    return a - Value::create(b);
}

ValuePtr operator-(double a, const ValuePtr& b) {
    return Value::create(a) - b;
}

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data * b->data, 
                                        std::vector<ValuePtr>{a, b}, 
                                        "*");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a, b]() {
        autograd::backward::mul(out_ptr, a, b);
    };
    return out;
}

ValuePtr operator*(const ValuePtr& a, double b) {
    return a * Value::create(b);
}

ValuePtr operator*(double a, const ValuePtr& b) {
    return Value::create(a) * b;
}

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data / b->data, 
                                        std::vector<ValuePtr>{a, b}, 
                                        "/");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a, b]() {
        autograd::backward::div(out_ptr, a, b);
    };
    return out;
}

ValuePtr operator/(const ValuePtr& a, double b) {
    return a / Value::create(b);
}

ValuePtr operator/(double a, const ValuePtr& b) {
    return Value::create(a) / b;
}

ValuePtr pow(const ValuePtr& a, double exponent) {
    auto out = std::make_shared<Value>(std::pow(a->data, exponent), 
                                        std::vector<ValuePtr>{a}, 
                                        "pow");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a, exponent]() {
        a->grad += exponent * std::pow(a->data, exponent - 1.0) * out_ptr->grad;
    };
    return out;
}

ValuePtr exp(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::exp(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "exp");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a]() {
        a->grad += out_ptr->data * out_ptr->grad;
    };
    return out;
}

ValuePtr log(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::log(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "log");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a]() {
        a->grad += (1.0 / a->data) * out_ptr->grad;
    };
    return out;
}

ValuePtr sqrt(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::sqrt(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "sqrt");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a]() {
        a->grad += (0.5 / out_ptr->data) * out_ptr->grad;
    };
    return out;
}

ValuePtr tanh(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::tanh(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "tanh");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a]() {
        autograd::backward::tanh(out_ptr, a);
    };
    return out;
}

ValuePtr relu(const ValuePtr& a) {
    auto out = std::make_shared<Value>(a->data > 0 ? a->data : 0.0, 
                                        std::vector<ValuePtr>{a}, 
                                        "relu");
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, a]() {
        autograd::backward::relu(out_ptr, a);
    };
    return out;
}

// HAPUS gelu dari sini - sudah ada di utils.cpp
// ValuePtr gelu(const ValuePtr& a) { ... }