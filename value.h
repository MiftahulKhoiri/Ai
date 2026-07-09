#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <string>

struct Value : std::enable_shared_from_this<Value> {
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Value>> _prev;
    std::string _op;

    Value(double data, std::vector<std::shared_ptr<Value>> children = {}, std::string op = "");
    static std::shared_ptr<Value> create(double data);
    void backward();
    std::string repr() const;
};

using ValuePtr = std::shared_ptr<Value>;

// Non-member operators
ValuePtr operator+(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator+(const ValuePtr& a, double b);
ValuePtr operator+(double a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, double b);
ValuePtr operator*(double a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, double b);
ValuePtr operator-(double a, const ValuePtr& b);
ValuePtr operator/(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator/(const ValuePtr& a, double b);
ValuePtr operator/(double a, const ValuePtr& b);
ValuePtr pow(const ValuePtr& a, double exponent);
ValuePtr exp(const ValuePtr& a);
ValuePtr log(const ValuePtr& a);
ValuePtr sqrt(const ValuePtr& a);
ValuePtr tanh(const ValuePtr& a);
ValuePtr relu(const ValuePtr& a);
ValuePtr gelu(const ValuePtr& a);