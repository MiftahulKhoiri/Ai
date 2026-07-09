#include "value.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <unordered_set>

Value::Value(double data, std::vector<std::shared_ptr<Value>> children, std::string op)
    : data(data), grad(0.0), _backward([]{}), _prev(children), _op(op) {}

std::shared_ptr<Value> Value::create(double data) {
    return std::make_shared<Value>(data);
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<Value*> visited;
    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
        if (visited.find(v.get()) != visited.end()) return;
        visited.insert(v.get());
        for (auto& child : v->_prev) build_topo(child);
        topo.push_back(v);
    };
    build_topo(shared_from_this());
    grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        (*it)->_backward();
}

std::string Value::repr() const {
    std::ostringstream oss;
    oss << "Value(data=" << data << ", grad=" << grad << ")";
    return oss.str();
}

// Operator implementations
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto out = Value::create(a->data + b->data);
    out->_prev = {a, b};
    out->_op = "+";
    out->_backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

ValuePtr operator+(const ValuePtr& a, double b) { return a + Value::create(b); }
ValuePtr operator+(double a, const ValuePtr& b) { return Value::create(a) + b; }

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    auto out = Value::create(a->data * b->data);
    out->_prev = {a, b};
    out->_op = "*";
    out->_backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

ValuePtr operator*(const ValuePtr& a, double b) { return a * Value::create(b); }
ValuePtr operator*(double a, const ValuePtr& b) { return Value::create(a) * b; }

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) { return a + (-1.0 * b); }
ValuePtr operator-(const ValuePtr& a, double b) { return a - Value::create(b); }
ValuePtr operator-(double a, const ValuePtr& b) { return Value::create(a) - b; }

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
    auto out = Value::create(a->data / b->data);
    out->_prev = {a, b};
    out->_op = "/";
    out->_backward = [a, b, out]() {
        a->grad += (1.0 / b->data) * out->grad;
        b->grad += (-a->data / (b->data * b->data)) * out->grad;
    };
    return out;
}

ValuePtr operator/(const ValuePtr& a, double b) { return a / Value::create(b); }
ValuePtr operator/(double a, const ValuePtr& b) { return Value::create(a) / b; }

ValuePtr pow(const ValuePtr& a, double exponent) {
    auto out = Value::create(std::pow(a->data, exponent));
    out->_prev = {a};
    out->_op = "**" + std::to_string(exponent);
    out->_backward = [a, exponent, out]() {
        a->grad += exponent * std::pow(a->data, exponent - 1) * out->grad;
    };
    return out;
}

ValuePtr exp(const ValuePtr& a) {
    double x = std::max(std::min(a->data, 60.0), -60.0);
    auto out = Value::create(std::exp(x));
    out->_prev = {a};
    out->_op = "exp";
    out->_backward = [a, out]() { a->grad += out->data * out->grad; };
    return out;
}

ValuePtr log(const ValuePtr& a) {
    double x = std::max(a->data, 1e-12);
    auto out = Value::create(std::log(x));
    out->_prev = {a};
    out->_op = "log";
    out->_backward = [a, x, out]() { a->grad += (1.0 / x) * out->grad; };
    return out;
}

ValuePtr sqrt(const ValuePtr& a) {
    double x = std::max(a->data, 1e-12);
    double r = std::sqrt(x);
    auto out = Value::create(r);
    out->_prev = {a};
    out->_op = "sqrt";
    out->_backward = [a, r, out]() { a->grad += (0.5 / r) * out->grad; };
    return out;
}

ValuePtr tanh(const ValuePtr& a) {
    double t = std::tanh(a->data);
    auto out = Value::create(t);
    out->_prev = {a};
    out->_op = "tanh";
    out->_backward = [a, t, out]() { a->grad += (1 - t * t) * out->grad; };
    return out;
}

ValuePtr relu(const ValuePtr& a) {
    double out_data = a->data > 0 ? a->data : 0.0;
    auto out = Value::create(out_data);
    out->_prev = {a};
    out->_op = "relu";
    out->_backward = [a, out]() { if (a->data > 0) a->grad += out->grad; };
    return out;
}

ValuePtr gelu(const ValuePtr& a) {
    double c = 0.7978845608028654;
    auto inner = (a + pow(a, 3) * 0.044715) * c;
    return (a * (tanh(inner) + 1.0)) * 0.5;
}