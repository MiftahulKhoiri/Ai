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
    // FIX: iteratif (bukan rekursif) untuk menghindari stack overflow
    // pada graph besar/dalam, dan pakai unordered_set<Value*> (bukan
    // std::find di vector) supaya lookup visited jadi O(1), bukan O(n).
    std::vector<Value::Ptr> topo;
    std::unordered_set<Value*> visited;

    // frame: {node, index child berikutnya yang mau diproses}
    std::vector<std::pair<Value::Ptr, size_t>> stack;
    stack.push_back({shared_from_this(), 0});
    visited.insert(this);

    while (!stack.empty()) {
        auto& [node, idx] = stack.back();

        if (idx < node->_prev.size()) {
            Value::Ptr child = node->_prev[idx];
            idx++; // maju ke child berikutnya untuk kunjungan selanjutnya
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

// Operator implementations
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto out = std::make_shared<Value>(a->data + b->data, 
                                        std::vector<ValuePtr>{a, b}, 
                                        "+");
    out->_backward = [out, a, b]() {
        autograd::backward::add(out.get(), a, b);
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
    out->_backward = [out, a, b]() {
        autograd::backward::sub(out.get(), a, b);
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
    out->_backward = [out, a, b]() {
        autograd::backward::mul(out.get(), a, b);
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
    out->_backward = [out, a, b]() {
        autograd::backward::div(out.get(), a, b);
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
    out->_backward = [out, a, exponent]() {
        a->grad += exponent * std::pow(a->data, exponent - 1.0) * out->grad;
    };
    return out;
}

ValuePtr exp(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::exp(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "exp");
    out->_backward = [out, a]() {
        a->grad += out->data * out->grad;
    };
    return out;
}

ValuePtr log(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::log(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "log");
    out->_backward = [out, a]() {
        a->grad += (1.0 / a->data) * out->grad;
    };
    return out;
}

ValuePtr sqrt(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::sqrt(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "sqrt");
    out->_backward = [out, a]() {
        a->grad += (0.5 / out->data) * out->grad;
    };
    return out;
}

ValuePtr tanh(const ValuePtr& a) {
    auto out = std::make_shared<Value>(std::tanh(a->data), 
                                        std::vector<ValuePtr>{a}, 
                                        "tanh");
    out->_backward = [out, a]() {
        autograd::backward::tanh(out.get(), a);
    };
    return out;
}

ValuePtr relu(const ValuePtr& a) {
    auto out = std::make_shared<Value>(a->data > 0 ? a->data : 0.0, 
                                        std::vector<ValuePtr>{a}, 
                                        "relu");
    out->_backward = [out, a]() {
        autograd::backward::relu(out.get(), a);
    };
    return out;
}

// HAPUS gelu dari sini - sudah ada di utils.cpp
// ValuePtr gelu(const ValuePtr& a) { ... }