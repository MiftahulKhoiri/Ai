#pragma once
#include "value.h"
#include <vector>

// Softmax: mengembalikan vektor probabilitas sebagai ValuePtr
std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits);