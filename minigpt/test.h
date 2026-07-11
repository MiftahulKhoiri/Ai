// test.h
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <cassert>

#define TEST_ASSERT(condition) \
    if (!(condition)) { \
        std::cerr << "❌ Test failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    }

#define TEST_ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "❌ Test failed: " << #a << " == " << #b << " (" << (a) << " != " << (b) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    }

#define TEST_ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "❌ Test failed: " << #a << " ≈ " << #b << " (" << (a) << " != " << (b) << ", tol=" << (tol) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    }

class TestRunner {
public:
    static TestRunner& instance() {
        static TestRunner runner;
        return runner;
    }
    
    void add_test(const std::string& name, std::function<bool()> test) {
        tests.emplace_back(name, test);
    }
    
    bool run_all() {
        int passed = 0, failed = 0;
        auto start = std::chrono::steady_clock::now();
        
        std::cout << "🧪 Running " << tests.size() << " tests..." << std::endl;
        std::cout << "========================================" << std::endl;
        
        for (const auto& [name, test] : tests) {
            std::cout << "▶ " << name << " ... ";
            auto test_start = std::chrono::steady_clock::now();
            
            try {
                if (test()) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - test_start);
                    std::cout << "✅ PASSED (" << duration.count() << "ms)" << std::endl;
                    passed++;
                } else {
                    std::cout << "❌ FAILED" << std::endl;
                    failed++;
                }
            } catch (const std::exception& e) {
                std::cout << "❌ EXCEPTION: " << e.what() << std::endl;
                failed++;
            } catch (...) {
                std::cout << "❌ UNKNOWN EXCEPTION" << std::endl;
                failed++;
            }
        }
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        
        std::cout << "========================================" << std::endl;
        std::cout << "📊 Results: " << passed << " passed, " << failed << " failed" << std::endl;
        std::cout << "⏱️  Time: " << duration.count() << "ms" << std::endl;
        
        return failed == 0;
    }
    
    void print_summary() {
        // Already printed in run_all
    }
    
private:
    std::vector<std::pair<std::string, std::function<bool()>>> tests;
};

// Macro for defining tests
#define TEST(name) \
    bool test_##name(); \
    struct TestRegistrar_##name { \
        TestRegistrar_##name() { \
            TestRunner::instance().add_test(#name, test_##name); \
        } \
    } test_registrar_##name; \
    bool test_##name()

// Example test
/*
TEST(ValueArithmetic) {
    auto v1 = Value::create(5.0);
    auto v2 = Value::create(3.0);
    auto v3 = v1 + v2;
    TEST_ASSERT_NEAR(v3->data, 8.0, 1e-6);
    return true;
}
*/