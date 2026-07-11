// backward.h
#pragma once
#include "value.h"
#include <memory>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace autograd {
namespace backward {

void add(Value* out, Value::Ptr a, Value::Ptr b);
void sub(Value* out, Value::Ptr a, Value::Ptr b);
void mul(Value* out, Value::Ptr a, Value::Ptr b);
void div(Value* out, Value::Ptr a, Value::Ptr b);
void tanh(Value* out, Value::Ptr a);
void relu(Value* out, Value::Ptr a);
void gelu(Value* out, Value::Ptr a);

} // namespace backward
} // namespace autograd