// test.cpp
#include "test.h"
#include "model.h"
#include "value.h"
#include "tokenizer.h"
#include <iostream>

// Contoh test untuk Value
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

TEST(ValueMultiplication) {
    auto v1 = Value::create(5.0);
    auto v2 = Value::create(3.0);
    auto v3 = v1 * v2;
    TEST_ASSERT_NEAR(v3->data, 15.0, 1e-6);
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

TEST(ModelCreation) {
    auto model = std::make_unique<MiniGPT>(100, 16, 2, 2, 32, 64);
    TEST_ASSERT(model != nullptr);
    TEST_ASSERT_EQ(model->d_model, 16);
    TEST_ASSERT_EQ(model->vocab_size, 100);
    TEST_ASSERT_EQ(model->max_len, 64);
    return true;
}

// Run all tests
bool run_all_tests() {
    return TestRunner::instance().run_all();
}

// If compiled as standalone
int main() {
    return run_all_tests() ? 0 : 1;
}