// value.cpp
#include "value.h"
#include <unordered_set>
#include <algorithm>

namespace autograd {

void Value::backward() {
  // Topological sort
  std::vector<Value*> order;
  std::unordered_set<Value*> visited;
  std::function<void(Value*)> build = [&](Value* v) {
    if (!visited.insert(v).second) return;
    for (auto& edge : v->prev) {
      if (auto p = edge.parent.lock()) {
        build(p.get());
      }
    }
    order.push_back(v);
  };
  build(this);

  this->grad = 1.0;
  for (int i = order.size()-1; i >= 0; --i) {
    if (order[i]->backward_fn) order[i]->backward_fn();
  }
}

} // namespace autograd