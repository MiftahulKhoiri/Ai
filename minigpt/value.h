// value.h
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>   // TAMBAHKAN

struct Value : std::enable_shared_from_this<Value> {
    using Ptr = std::shared_ptr<Value>;  // TAMBAHKAN INI!
    
    double data;
    double grad;

    std::function<void()> _backward;
    std::vector<Value::Ptr> _prev;
    std::string _op;

    explicit Value(
        double data,
        std::vector<Value::Ptr> children = {},
        std::string op = ""
    );

    static Value::Ptr create(double data);

    void backward();
    std::string repr() const;
};

using ValuePtr = std::shared_ptr<Value>;

// Arithmetic operators
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator+(const ValuePtr& a, double b);
ValuePtr operator+(double a, const ValuePtr& b);

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, double b);
ValuePtr operator-(double a, const ValuePtr& b);

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, double b);
ValuePtr operator*(double a, const ValuePtr& b);

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator/(const ValuePtr& a, double b);
ValuePtr operator/(double a, const ValuePtr& b);

// Math functions
ValuePtr pow(const ValuePtr& a, double exponent);
ValuePtr exp(const ValuePtr& a);
ValuePtr log(const ValuePtr& a);
ValuePtr sqrt(const ValuePtr& a);
ValuePtr tanh(const ValuePtr& a);
ValuePtr relu(const ValuePtr& a);
ValuePtr gelu(const ValuePtr& a);