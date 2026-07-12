// utils.cpp
#include "utils.h"
#include "backward.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ValuePtr gelu(const ValuePtr& x) {
    double val = x->data;
    double c = std::sqrt(2.0 / M_PI);
    double x3 = val * val * val;
    double tanh_arg = c * (val + 0.044715 * x3);
    double result = 0.5 * val * (1.0 + std::tanh(tanh_arg));

    auto out = Value::create(result);
    out->_prev = {x};
    out->_op = "gelu";

    // FIX: capture raw pointer (out_ptr), BUKAN shared_ptr "out" itu sendiri.
    // Capture "out" langsung akan membuat reference cycle:
    // out -> _backward (closure) -> capture out -> kembali ke out.
    // Refcount out tidak akan pernah turun ke 0 -> node bocor permanen.
    Value* out_ptr = out.get();
    out->_backward = [out_ptr, x]() {
        // Reuse turunan GELU yang sudah ada di backward.cpp,
        // supaya rumus tidak terduplikasi di dua tempat.
        autograd::backward::gelu(out_ptr, x);
    };

    return out;
}