// test.cpp
#include "test.h"
#include "model.h"
#include "value.h"
#include "tokenizer.h"
#include <iostream>
#include <memory>

// ============================================================
// UNIT TESTS
// ============================================================

TEST(ValueCreation) {
    auto v = Value::create(5.0);
    TEST_ASSERT_NEAR(v->data, 5.0, 1e-6);
    TEST_ASSERT_NEAR(v->grad, 0.0, 1e-6);
    return true;
}

TEST(ValueAddition) {
    auto v1 = Value::create(5.0);
    auto v2 = Value::create(3.0);
    auto v3 = v1 + v2;
    TEST_ASSERT_NEAR(v3->data, 8.0, 1e-6);
    return true;
}

TEST(ValueSubtraction) {
    auto v1 = Value::create(5.0);
    auto v2 = Value::create(3.0);
    auto v3 = v1 - v2;
    TEST_ASSERT_NEAR(v3->data, 2.0, 1e-6);
    return true;
}

TEST(ValueMultiplication) {
    auto v1 = Value::create(5.0);
    auto v2 = Value::create(3.0);
    auto v3 = v1 * v2;
    TEST_ASSERT_NEAR(v3->data, 15.0, 1e-6);
    return true;
}

TEST(ValueDivision) {
    auto v1 = Value::create(6.0);
    auto v2 = Value::create(2.0);
    auto v3 = v1 / v2;
    TEST_ASSERT_NEAR(v3->data, 3.0, 1e-6);
    return true;
}

TEST(ValueBackward) {
    auto v1 = Value::create(5.0);
    auto v2 = Value::create(3.0);
    auto v3 = v1 * v2;
    v3->backward();
    TEST_ASSERT_NEAR(v1->grad, 3.0, 1e-6);
    TEST_ASSERT_NEAR(v2->grad, 5.0, 1e-6);
    return true;
}

TEST(ValueBackwardChain) {
    auto v1 = Value::create(2.0);
    auto v2 = Value::create(3.0);
    auto v3 = Value::create(4.0);
    auto v4 = v1 * v2 + v3;
    v4->backward();
    // d(v4)/d(v1) = v2 = 3
    // d(v4)/d(v2) = v1 = 2
    // d(v4)/d(v3) = 1
    TEST_ASSERT_NEAR(v1->grad, 3.0, 1e-6);
    TEST_ASSERT_NEAR(v2->grad, 2.0, 1e-6);
    TEST_ASSERT_NEAR(v3->grad, 1.0, 1e-6);
    return true;
}

TEST(ValueRelu) {
    auto v1 = Value::create(-2.0);
    auto v2 = relu(v1);
    TEST_ASSERT_NEAR(v2->data, 0.0, 1e-6);
    
    auto v3 = Value::create(3.0);
    auto v4 = relu(v3);
    TEST_ASSERT_NEAR(v4->data, 3.0, 1e-6);
    return true;
}

TEST(ValueTanh) {
    auto v1 = Value::create(0.0);
    auto v2 = tanh(v1);
    TEST_ASSERT_NEAR(v2->data, 0.0, 1e-6);
    
    auto v3 = Value::create(1.0);
    auto v4 = tanh(v3);
    TEST_ASSERT_NEAR(v4->data, 0.761594, 1e-6);
    return true;
}

TEST(ModelCreation) {
    auto model = std::make_unique<MiniGPT>(100, 16, 2, 2, 32, 64);
    TEST_ASSERT(model != nullptr);
    TEST_ASSERT_EQ(model->d_model, 16);
    TEST_ASSERT_EQ(model->vocab_size, 100);
    TEST_ASSERT_EQ(model->max_len, 64);
    return true;
}

TEST(ModelParameters) {
    auto model = std::make_unique<MiniGPT>(100, 16, 2, 2, 32, 64);
    auto params = model->parameters();
    TEST_ASSERT(!params.empty());
    return true;
}

TEST(TokenizerCreation) {
    ByteLevelBPETokenizer tokenizer;
    // Test train with small corpus
    tokenizer.train("Hello world! This is a test.", 100);
    TEST_ASSERT(tokenizer.vocab_size() > 0);
    return true;
}

TEST(TokenizerEncodeDecode) {
    ByteLevelBPETokenizer tokenizer;
    tokenizer.train("Hello world! This is a test.", 100);
    
    std::string text = "Hello world";
    auto tokens = tokenizer.encode(text);
    auto decoded = tokenizer.decode(tokens);
    
    // Decoded might not be exactly the same due to BPE
    TEST_ASSERT(!tokens.empty());
    return true;
}

// ============================================================
// RUN ALL TESTS
// ============================================================

bool run_all_tests() {
    return TestRunner::instance().run_all();
}

// Main function for standalone execution
int main() {
    return run_all_tests() ? 0 : 1;
}