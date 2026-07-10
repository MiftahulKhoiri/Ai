#include "utils.h"
#include <algorithm>
#include <limits>

std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits) {
    // Tangani input kosong
    if (logits.empty()) {
        return {};
    }

    // Cari nilai maksimum untuk numerical stability
    auto max_it = std::max_element(
        logits.begin(),
        logits.end(),
        [](const ValuePtr& a, const ValuePtr& b) {
            return a->data < b->data;
        });

    double max_data = (*max_it)->data;

    // Hitung exp(logit - max)
    std::vector<ValuePtr> exps;
    exps.reserve(logits.size());

    for (const auto& v : logits) {
        exps.push_back(exp(v - max_data));
    }

    // Jumlahkan semua nilai eksponensial
    ValuePtr sum = Value::create(0.0);

    for (const auto& e : exps) {
        sum = sum + e;
    }

    // Normalisasi menjadi probabilitas
    std::vector<ValuePtr> probs;
    probs.reserve(logits.size());

    for (const auto& e : exps) {
        probs.push_back(e / sum);
    }

    return probs;
}