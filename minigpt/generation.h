// generation.h
#pragma once
#include "model.h"
#include "tokenizer.h"
#include <string>
#include <vector>

void generate(MiniGPT& model, Tokenizer& tokenizer,
              const std::string& prompt, int max_tokens = 50);

// Fungsi pembantu
int argmax(const std::vector<Value::Ptr>& logits);
std::vector<int> sample_top_k(const std::vector<Value::Ptr>& logits, int k = 0, float temperature = 1.0);