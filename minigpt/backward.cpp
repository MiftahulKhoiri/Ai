// backward.cpp
#include "backward.h"

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
    // Saat forward: out = a * b, maka d(out)/da = b, d(out)/db = a
    // b dan a diambil dari data saat forward (tersimpan di Edge::gradient_factor)
    // Pada lambda kita akan memanggil fungsi ini dengan a, b yang masih hidup
    a->grad += out->_prev[1].gradient_factor * out->grad;  // b
    b->grad += out->_prev[0].gradient_factor * out->grad;  // a
}

void div(Value* out, Value::Ptr a, Value::Ptr b) {
    // out = a/b,  da = 1/b, db = -a/(b*b)
    a->grad += (1.0 / b->data) * out->grad;
    b->grad += (-a->data / (b->data * b->data)) * out->grad;
}

void tanh(Value* out, Value::Ptr a) {
    double t = out->data;
    a->grad += (1.0 - t * t) * out->grad;
}

void relu(Value* out, Value::Ptr a) {
    a->grad += (a->data > 0 ? 1.0 : 0.0) * out->grad;
}

} // namespace backward
} // namespace autograd