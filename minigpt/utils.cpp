// utils.cpp
#include "utils.h"
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
    
    out->_backward = [out, x]() {
        double grad = out->grad;
        double v = x->data;
        double c = std::sqrt(2.0 / M_PI);
        double v3 = v * v * v;
        double tanh_arg = c * (v + 0.044715 * v3);
        double tanh_val = std::tanh(tanh_arg);
        double sech2 = 1.0 - tanh_val * tanh_val;
        double dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * v * c * (1.0 + 3.0 * 0.044715 * v * v) * sech2;
        x->grad += dgelu * grad;
    };
    
    return out;
}