// graph.h
#pragma once
#include "value.h"
#include <vector>
#include <memory>

namespace autograd {

class Graph {
public:
    void add_node(Value::Ptr node);
    void zero_grad();
    const std::vector<Value::Ptr>& nodes() const { return _nodes; }

private:
    std::vector<Value::Ptr> _nodes;
};

} // namespace autograd