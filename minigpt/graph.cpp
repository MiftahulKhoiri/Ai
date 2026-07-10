// graph.cpp
#include "graph.h"

namespace autograd {

void Graph::add_node(Value::Ptr node) {
    _nodes.push_back(node);
}

void Graph::zero_grad() {
    for (auto& node : _nodes) {
        node->zero_grad();
    }
}

} // namespace autograd