#pragma once
#include <string>
#include "model.h"
#include "tokenizer.h"

int sample_from_logits(const std::vector<ValuePtr>& logits, double temperature, int top_k, double top_p);
std::string generate(MiniGPT& model, ByteLevelBPETokenizer& tok, const std::string& prompt,
                     int max_new_tokens = 25, double temperature = 0.9, int top_k = 0, double top_p = 0.9);