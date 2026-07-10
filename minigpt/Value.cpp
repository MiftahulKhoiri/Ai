// value.cpp
#include "value.h"
#include "backward.h"
#include <algorithm>
#include <cmath>

namespace autograd {

Value::Ptr Value::create(double data) {
    return Ptr(new Value(data));
}

void Value::zero_grad() {
    grad = 0.0;
}

// ----- Operator overloading -----
Value::Ptr Value::operator+(Ptr other) {
    auto out = create(data + other->data);
    auto weak_this = weak_from_this();
    auto weak_other = other->weak_from_this();
    out->_prev.push_back({weak_this, 1.0});
    out->_prev.push_back({weak_other, 1.0});
    out->_backward_fn = [weak_this, weak_other, out_ptr = out.get()]() {
        auto a = weak_this.lock();
        auto b = weak_other.lock();
        if (a && b) backward::add(out_ptr, a, b);
    };
    return out;
}

Value::Ptr Value::operator-(Ptr other) {
    auto out = create(data - other->data);
    auto weak_this = weak_from_this();
    auto weak_other = other->weak_from_this();
    out->_prev.push_back({weak_this, 1.0});
    out->_prev.push_back({weak_other, -1.0});
    out->_backward_fn = [weak_this, weak_other, out_ptr = out.get()]() {
        auto a = weak_this.lock();
        auto b = weak_other.lock();
        if (a && b) backward::sub(out_ptr, a, b);
    };
    return out;
}

Value::Ptr Value::operator*(Ptr other) {
    auto out = create(data * other->data);
    auto weak_this = weak_from_this();
    auto weak_other = other->weak_from_this();
    out->_prev.push_back({weak_this, other->data});   // d(out)/d(this) = other->data
    out->_prev.push_back({weak_other, data});         // d(out)/d(other) = this->data
    out->_backward_fn = [weak_this, weak_other, out_ptr = out.get()]() {
        auto a = weak_this.lock();
        auto b = weak_other.lock();
        if (a && b) backward::mul(out_ptr, a, b);
    };
    return out;
}

Value::Ptr Value::operator/(Ptr other) {
    auto out = create(data / other->data);
    auto weak_this = weak_from_this();
    auto weak_other = other->weak_from_this();
    out->_prev.push_back({weak_this, 1.0 / other->data});
    out->_prev.push_back({weak_other, -data / (other->data * other->data)});
    out->_backward_fn = [weak_this, weak_other, out_ptr = out.get()]() {
        auto a = weak_this.lock();
        auto b = weak_other.lock();
        if (a && b) backward::div(out_ptr, a, b);
    };
    return out;
}

Value::Ptr Value::tanh() {
    double t = std::tanh(data);
    auto out = create(t);
    auto weak_this = weak_from_this();
    out->_prev.push_back({weak_this, 1.0 - t*t});
    out->_backward_fn = [weak_this, out_ptr = out.get()]() {
        auto a = weak_this.lock();
        if (a) backward::tanh(out_ptr, a);
    };
    return out;
}

Value::Ptr Value::relu() {
    double r = data > 0 ? data : 0.0;
    auto out = create(r);
    auto weak_this = weak_from_this();
    out->_prev.push_back({weak_this, data > 0 ? 1.0 : 0.0});
    out->_backward_fn = [weak_this, out_ptr = out.get()]() {
        auto a = weak_this.lock();
        if (a) backward::relu(out_ptr, a);
    };
    return out;
}

// Topological sort
void Value::_build_topo(std::vector<Value*>& order, std::unordered_set<Value*>& visited) {
    if (visited.find(this) != visited.end()) return;
    visited.insert(this);
    for (auto& edge : _prev) {
        if (auto p = edge.parent.lock()) {
            p->_build_topo(order, visited);
        }
    }
    order.push_back(this);
}

// Backward pass
void Value::backward() {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    _build_topo(topo, visited);
    
    this->grad = 1.0;               // seed gradient on loss
    // Reverse topological order
    for (int i = topo.size() - 1; i >= 0; --i) {
        if (topo[i]->_backward_fn) {
            topo[i]->_backward_fn();
        }
    }
}

} // namespace autograd