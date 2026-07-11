// backward.cpp
#include "backward.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace autograd {
namespace backward {

void add(Value* out, Value::Ptr a, Value::Ptr b) {
    a->grad += out->grad;
    b->grad += out->grad;
}

void sub(Value* out, Value::Ptr a, Value::Ptr b) {
    a->grad += out->grad;
    b->grad += -out->grad;
}

void mul(Value* out, Value::Ptr a, Value::Ptr b) {
    // out = a * b
    // d(out)/da = b, d(out)/db = a
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
}

void div(Value* out, Value::Ptr a, Value::Ptr b) {
    // out = a / b
    // d(out)/da = 1/b, d(out)/db = -a/(b*b)
    a->grad += (1.0 / b->data) * out->grad;
    b->grad += (-a->data / (b->data * b->data)) * out->grad;
}

void tanh(Value* out, Value::Ptr a) {
    // out = tanh(a)
    // d(out)/da = 1 - tanh^2(a) = 1 - out^2
    double t = out->data;
    a->grad += (1.0 - t * t) * out->grad;
}

void relu(Value* out, Value::Ptr a) {
    // out = relu(a) = max(0, a)
    // d(out)/da = 1 if a > 0 else 0
    a->grad += (a->data > 0 ? 1.0 : 0.0) * out->grad;
}

void gelu(Value* out, Value::Ptr a) {
    // out = GELU(a) ≈ 0.5 * a * (1 + tanh(sqrt(2/π) * (a + 0.044715 * a^3)))
    // Approximate derivative
    double x = a->data;
    double c = std::sqrt(2.0 / M_PI);
    double x3 = x * x * x;
    double tanh_arg = c * (x + 0.044715 * x3);
    double tanh_val = std::tanh(tanh_arg);
    double sech2 = 1.0 - tanh_val * tanh_val;
    double dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * c * (1.0 + 3.0 * 0.044715 * x * x) * sech2;
    a->grad += dgelu * out->grad;
}

} // namespace backward
} // namespace autograd