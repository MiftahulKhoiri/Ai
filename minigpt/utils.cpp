#include "utils.h"
#include <algorithm>

std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits) {
    double max_data = -1e30;
    for (auto& v : logits)
        if (v->data > max_data) max_data = v->data;
    std::vector<ValuePtr> exps;
    for (auto& v : logits)
        exps.push_back(exp(v - max_data));
    ValuePtr sum = Value::create(0.0);
    for (auto& e : exps)
        sum = sum + e;
    std::vector<ValuePtr> probs;
    for (auto& e : exps)
        probs.push_back(e / sum);
    return probs;
}