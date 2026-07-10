// value.h
#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <cmath>
#include <memory_resource>
#include <iostream>

namespace autograd {

class Value : public std::enable_shared_from_this<Value> {
public:
  double data;
  double grad{0.0};

  using Ptr = std::shared_ptr<Value>;

  // Factory
  static Ptr create(double data) {
    return Ptr(new Value(data));
  }

  // Operasi dengan graph
  Ptr operator+(Ptr other);
  Ptr operator-(Ptr other);
  Ptr operator*(Ptr other);
  Ptr operator/(Ptr other);
  Ptr tanh();
  Ptr relu();

  void backward();

private:
  Value(double d) : data(d) {}
  // Struktur untuk backward
  struct Edge {
    std::weak_ptr<Value> parent;
    double gradient_factor; // d(child)/d(parent)
  };
  std::vector<Edge> prev;
  std::function<void()> backward_fn;

  void topological_sort(std::vector<Value*>& order, std::unordered_set<Value*>& visited);
};

// Inline implementations for common ops
inline Value::Ptr Value::operator+(Ptr other) {
  auto out = create(this->data + other->data);
  auto weak_this = this->weak_from_this();
  auto weak_other = other->weak_from_this();
  out->prev.push_back({weak_this, 1.0});
  out->prev.push_back({weak_other, 1.0});
  out->backward_fn = [weak_this, weak_other, out_ptr = out.get()]() {
    if (auto t = weak_this.lock()) t->grad += 1.0 * out_ptr->grad;
    if (auto o = weak_other.lock()) o->grad += 1.0 * out_ptr->grad;
  };
  return out;
}

inline Value::Ptr Value::operator-(Ptr other) {
  auto out = create(this->data - other->data);
  auto weak_this = this->weak_from_this();
  auto weak_other = other->weak_from_this();
  out->prev.push_back({weak_this, 1.0});
  out->prev.push_back({weak_other, -1.0});
  out->backward_fn = [weak_this, weak_other, out_ptr = out.get()]() {
    if (auto t = weak_this.lock()) t->grad += 1.0 * out_ptr->grad;
    if (auto o = weak_other.lock()) o->grad += -1.0 * out_ptr->grad;
  };
  return out;
}

inline Value::Ptr Value::operator*(Ptr other) {
  auto out = create(this->data * other->data);
  double a = this->data, b = other->data;
  auto weak_this = this->weak_from_this();
  auto weak_other = other->weak_from_this();
  out->prev.push_back({weak_this, b});
  out->prev.push_back({weak_other, a});
  out->backward_fn = [weak_this, weak_other, out_ptr = out.get(), b, a]() {
    if (auto t = weak_this.lock()) t->grad += b * out_ptr->grad;
    if (auto o = weak_other.lock()) o->grad += a * out_ptr->grad;
  };
  return out;
}

inline Value::Ptr Value::operator/(Ptr other) {
  auto out = create(this->data / other->data);
  double a = this->data, b = other->data;
  auto weak_this = this->weak_from_this();
  auto weak_other = other->weak_from_this();
  out->prev.push_back({weak_this, 1.0/b});
  out->prev.push_back({weak_other, -a/(b*b)});
  out->backward_fn = [weak_this, weak_other, out_ptr = out.get(), b, a]() {
    if (auto t = weak_this.lock()) t->grad += (1.0/b) * out_ptr->grad;
    if (auto o = weak_other.lock()) o->grad += (-a/(b*b)) * out_ptr->grad;
  };
  return out;
}

inline Value::Ptr Value::tanh() {
  double t = std::tanh(this->data);
  auto out = create(t);
  auto weak_this = this->weak_from_this();
  out->prev.push_back({weak_this, 1.0 - t*t});
  out->backward_fn = [weak_this, out_ptr = out.get(), t]() {
    if (auto thiz = weak_this.lock()) thiz->grad += (1.0 - t*t) * out_ptr->grad;
  };
  return out;
}

inline Value::Ptr Value::relu() {
  double val = this->data > 0 ? this->data : 0.0;
  auto out = create(val);
  double local_grad = this->data > 0 ? 1.0 : 0.0;
  auto weak_this = this->weak_from_this();
  out->prev.push_back({weak_this, local_grad});
  out->backward_fn = [weak_this, out_ptr = out.get(), local_grad]() {
    if (auto t = weak_this.lock()) t->grad += local_grad * out_ptr->grad;
  };
  return out;
}

} // namespace autograd