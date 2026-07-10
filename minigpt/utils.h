#pragma once
#include <vector>
#include "value.h"
std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits);