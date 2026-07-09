#pragma once
#include "value.h"
#include <vector>
std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits);