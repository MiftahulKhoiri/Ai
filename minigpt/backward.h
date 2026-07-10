// backward.h
#pragma once
#include "value.h"

namespace autograd {
namespace backward {

void add(Value* out, Value::Ptr a, Value::Ptr b);
void sub(Value* out, Value::Ptr a, Value::Ptr b);
void mul(Value* out, Value::Ptr a, Value::Ptr b);
void div(Value* out, Value::Ptr a, Value::Ptr b);
void tanh(Value* out, Value::Ptr a);
void relu(Value* out, Value::Ptr a);

} // namespace backward
} // namespace autograd