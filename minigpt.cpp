#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <regex>
#include <sstream>
#include <fstream>
#include <iostream>
#include <functional>
#include <limits>
#include <numeric>

namespace py = pybind11;

// ============================================================
// 1. AUTOGRAD ENGINE: Value
// ============================================================
struct Value : std::enable_shared_from_this<Value> {
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Value>> _prev;
    std::string _op;

    Value(double data, std::vector<std::shared_ptr<Value>> children = {}, std::string op = "")
        : data(data), grad(0.0), _backward([]{}), _prev(children), _op(op) {}

    static std::shared_ptr<Value> create(double data) {
        return std::make_shared<Value>(data);
    }

    void backward() {
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<Value*> visited;
        std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
            if (visited.find(v.get()) != visited.end()) return;
            visited.insert(v.get());
            for (auto& child : v->_prev) build_topo(child);
            topo.push_back(v);
        };
        build_topo(shared_from_this());
        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward();
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Value(data=" << data << ", grad=" << grad << ")";
        return oss.str();
    }
};

// Non-member operators for shared_ptr<Value>
using ValuePtr = std::shared_ptr<Value>;

inline ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
    auto out = Value::create(a->data + b->data);
    out->_prev = {a, b};
    out->_op = "+";
    out->_backward = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

inline ValuePtr operator+(const ValuePtr& a, double b) {
    return a + Value::create(b);
}

inline ValuePtr operator+(double a, const ValuePtr& b) {
    return Value::create(a) + b;
}

inline ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
    auto out = Value::create(a->data * b->data);
    out->_prev = {a, b};
    out->_op = "*";
    out->_backward = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    return out;
}

inline ValuePtr operator*(const ValuePtr& a, double b) {
    return a * Value::create(b);
}

inline ValuePtr operator*(double a, const ValuePtr& b) {
    return Value::create(a) * b;
}

inline ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) {
    return a + (-1.0 * b);
}

inline ValuePtr operator-(const ValuePtr& a, double b) {
    return a - Value::create(b);
}

inline ValuePtr operator-(double a, const ValuePtr& b) {
    return Value::create(a) - b;
}

inline ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
    auto out = Value::create(a->data / b->data);
    out->_prev = {a, b};
    out->_op = "/";
    out->_backward = [a, b, out]() {
        a->grad += (1.0 / b->data) * out->grad;
        b->grad += (-a->data / (b->data * b->data)) * out->grad;
    };
    return out;
}

inline ValuePtr operator/(const ValuePtr& a, double b) {
    return a / Value::create(b);
}

inline ValuePtr operator/(double a, const ValuePtr& b) {
    return Value::create(a) / b;
}

inline ValuePtr pow(const ValuePtr& a, double exponent) {
    auto out = Value::create(std::pow(a->data, exponent));
    out->_prev = {a};
    out->_op = "**" + std::to_string(exponent);
    out->_backward = [a, exponent, out]() {
        a->grad += exponent * std::pow(a->data, exponent - 1) * out->grad;
    };
    return out;
}

inline ValuePtr exp(const ValuePtr& a) {
    double x = std::max(std::min(a->data, 60.0), -60.0);
    auto out = Value::create(std::exp(x));
    out->_prev = {a};
    out->_op = "exp";
    out->_backward = [a, out]() { a->grad += out->data * out->grad; };
    return out;
}

inline ValuePtr log(const ValuePtr& a) {
    double x = std::max(a->data, 1e-12);
    auto out = Value::create(std::log(x));
    out->_prev = {a};
    out->_op = "log";
    out->_backward = [a, x, out]() { a->grad += (1.0 / x) * out->grad; };
    return out;
}

inline ValuePtr sqrt(const ValuePtr& a) {
    double x = std::max(a->data, 1e-12);
    double r = std::sqrt(x);
    auto out = Value::create(r);
    out->_prev = {a};
    out->_op = "sqrt";
    out->_backward = [a, r, out]() { a->grad += (0.5 / r) * out->grad; };
    return out;
}

inline ValuePtr tanh(const ValuePtr& a) {
    double t = std::tanh(a->data);
    auto out = Value::create(t);
    out->_prev = {a};
    out->_op = "tanh";
    out->_backward = [a, t, out]() { a->grad += (1 - t * t) * out->grad; };
    return out;
}

inline ValuePtr relu(const ValuePtr& a) {
    double out_data = a->data > 0 ? a->data : 0.0;
    auto out = Value::create(out_data);
    out->_prev = {a};
    out->_op = "relu";
    out->_backward = [a, out]() { if (a->data > 0) a->grad += out->grad; };
    return out;
}

inline ValuePtr gelu(const ValuePtr& a) {
    double c = 0.7978845608028654;
    auto inner = (a + pow(a, 3) * 0.044715) * c;
    return (a * (tanh(inner) + 1.0)) * 0.5;
}

// ============================================================
// 2. TOKENIZER BYTE-LEVEL BPE
// ============================================================
static std::unordered_map<int, std::string> BYTE_ENCODER;
static std::unordered_map<std::string, int> BYTE_DECODER;

void init_byte_maps() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xa1; b <= 0xac; ++b) bs.push_back(b);
    for (int b = 0xae; b <= 0xff; ++b) bs.push_back(b);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            BYTE_ENCODER[b] = std::string(1, static_cast<char>(256 + n));
            ++n;
        } else {
            BYTE_ENCODER[b] = std::string(1, static_cast<char>(b));
        }
    }
    for (auto& p : BYTE_ENCODER) BYTE_DECODER[p.second] = p.first;
}

static const std::regex PRETOKEN_PAT(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)"
);

struct ByteLevelBPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::vector<std::pair<std::string, std::string>> merge_order;
    std::unordered_map<std::pair<std::string, std::string>, int, boost::hash<std::pair<std::string, std::string>>> merge_rank;

    void train(const std::string& corpus, int vocab_size = 400) {
        init_byte_maps();
        // Tokenisasi kata menjadi simbol byte-level
        std::vector<std::vector<std::string>> words;
        std::sregex_iterator it(corpus.begin(), corpus.end(), PRETOKEN_PAT);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            std::string token = it->str();
            std::vector<std::string> syms;
            for (unsigned char c : token)
                syms.push_back(BYTE_ENCODER.at(c));
            words.push_back(syms);
        }

        std::map<std::vector<std::string>, int> freq;
        for (auto& w : words) freq[w]++;

        std::map<std::vector<std::string>, std::vector<std::string>> splits;
        for (auto& p : freq) splits[p.first] = p.first;

        std::set<std::string> base_vocab;
        for (auto& p : BYTE_ENCODER) base_vocab.insert(p.second);

        int num_merges = std::max(0, vocab_size - (int)base_vocab.size() - 4);
        for (int step = 0; step < num_merges; ++step) {
            std::map<std::pair<std::string, std::string>, int> pair_counts;
            for (auto& wf : freq) {
                auto& syms = splits[wf.first];
                for (size_t i = 0; i + 1 < syms.size(); ++i) {
                    pair_counts[{syms[i], syms[i+1]}] += wf.second;
                }
            }
            if (pair_counts.empty()) break;
            auto best = std::max_element(pair_counts.begin(), pair_counts.end(),
                [](auto& a, auto& b) { return a.second < b.second; });
            if (best->second < 2) break;
            std::pair<std::string, std::string> best_pair = best->first;
            std::string merged = best_pair.first + best_pair.second;
            merge_order.push_back(best_pair);

            for (auto& wf : freq) {
                auto& syms = splits[wf.first];
                std::vector<std::string> new_syms;
                for (size_t i = 0; i < syms.size(); ) {
                    if (i + 1 < syms.size() && syms[i] == best_pair.first && syms[i+1] == best_pair.second) {
                        new_syms.push_back(merged);
                        i += 2;
                    } else {
                        new_syms.push_back(syms[i]);
                        i++;
                    }
                }
                splits[wf.first] = new_syms;
            }
        }

        std::vector<std::string> specials = {"<pad>", "<bos>", "<eos>", "<unk>"};
        std::vector<std::string> all_tokens = specials;
        for (auto& s : base_vocab) all_tokens.push_back(s);
        for (auto& p : merge_order) all_tokens.push_back(p.first + p.second);
        std::sort(all_tokens.begin() + 4, all_tokens.end());
        all_tokens.erase(std::unique(all_tokens.begin() + 4, all_tokens.end()), all_tokens.end());
        for (size_t i = 0; i < all_tokens.size(); ++i) {
            vocab[all_tokens[i]] = i;
            inv_vocab[i] = all_tokens[i];
        }
        for (size_t i = 0; i < merge_order.size(); ++i)
            merge_rank[merge_order[i]] = i;
    }

    std::vector<std::string> apply_bpe(const std::vector<std::string>& symbols) {
        std::vector<std::string> word = symbols;
        while (word.size() > 1) {
            std::pair<std::string, std::string> best_pair;
            int best_rank = std::numeric_limits<int>::max();
            bool found = false;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_rank.find({word[i], word[i+1]});
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_pair = it->first;
                    best_rank = it->second;
                    found = true;
                }
            }
            if (!found) break;
            std::string merged = best_pair.first + best_pair.second;
            std::vector<std::string> new_word;
            size_t i = 0;
            while (i < word.size()) {
                if (i + 1 < word.size() && word[i] == best_pair.first && word[i+1] == best_pair.second) {
                    new_word.push_back(merged);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i++;
                }
            }
            word = new_word;
        }
        return word;
    }

    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false) {
        std::vector<int> ids;
        if (add_bos) ids.push_back(vocab["<bos>"]);
        std::sregex_iterator it(text.begin(), text.end(), PRETOKEN_PAT);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            std::string token = it->str();
            std::vector<std::string> syms;
            for (unsigned char c : token) syms.push_back(BYTE_ENCODER.at(c));
            for (auto& s : apply_bpe(syms)) {
                auto vit = vocab.find(s);
                ids.push_back(vit != vocab.end() ? vit->second : vocab["<unk>"]);
            }
        }
        if (add_eos) ids.push_back(vocab["<eos>"]);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string bytes;
        for (int id : ids) {
            auto it = inv_vocab.find(id);
            if (it != inv_vocab.end()) {
                std::string tok = it->second;
                if (tok == "<pad>" || tok == "<bos>" || tok == "<eos>" || tok == "<unk>")
                    continue;
                for (char c : tok)
                    if (BYTE_DECODER.count(std::string(1,c)))
                        bytes.push_back(static_cast<char>(BYTE_DECODER[std::string(1,c)]));
            }
        }
        return bytes; // as UTF-8
    }

    void save(const std::string& path) {
        py::dict data;
        data["vocab"] = vocab;
        std::vector<py::list> mo;
        for (auto& p : merge_order) {
            py::list t;
            t.append(p.first);
            t.append(p.second);
            mo.push_back(t);
        }
        data["merge_order"] = mo;
        py::module_ json = py::module_::import("json");
        std::string dumped = json.attr("dumps")(data).cast<std::string>();
        std::ofstream f(path);
        f << dumped;
    }

    void load(const std::string& path) {
        py::module_ json = py::module_::import("json");
        std::ifstream f(path);
        std::stringstream buffer;
        buffer << f.rdbuf();
        py::dict data = json.attr("loads")(buffer.str()).cast<py::dict>();
        vocab = data["vocab"].cast<std::unordered_map<std::string, int>>();
        inv_vocab.clear();
        for (auto& p : vocab) inv_vocab[p.second] = p.first;
        merge_order.clear();
        merge_rank.clear();
        std::vector<py::list> mo = data["merge_order"].cast<std::vector<py::list>>();
        for (auto& t : mo) {
            std::string a = t[0].cast<std::string>();
            std::string b = t[1].cast<std::string>();
            merge_order.push_back({a, b});
        }
        for (size_t i = 0; i < merge_order.size(); ++i)
            merge_rank[merge_order[i]] = i;
    }
};

// ============================================================
// 3. UTILITY FUNCTIONS (Vector, Matrix, Softmax)
// ============================================================
ValuePtr dot(const std::vector<ValuePtr>& a, const std::vector<ValuePtr>& b) {
    ValuePtr s = Value::create(0.0);
    for (size_t i = 0; i < a.size(); ++i) s = s + a[i] * b[i];
    return s;
}

std::vector<ValuePtr> softmax(const std::vector<ValuePtr>& logits) {
    double m = std::max_element(logits.begin(), logits.end(),
        [](const ValuePtr& a, const ValuePtr& b) { return a->data < b->data; })->get()->data;
    std::vector<ValuePtr> exps;
    for (auto& x : logits) exps.push_back(exp(x - m));
    ValuePtr total = Value::create(0.0);
    for (auto& e : exps) total = total + e;
    std::vector<ValuePtr> probs;
    for (auto& e : exps) probs.push_back(e / total);
    return probs;
}

// ============================================================
// 4. DROPOUT
// ============================================================
struct Dropout {
    double p;
    bool training;
    std::mt19937 rng;

    Dropout(double p = 0.1) : p(p), training(true), rng(std::random_device{}()) {}

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) {
        if (!training || p <= 0) return x;
        double keep = 1.0 - p;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<ValuePtr> out;
        for (auto& v : x) {
            if (dist(rng) < keep)
                out.push_back(v / keep);
            else
                out.push_back(v * 0.0);
        }
        return out;
    }
};

// ============================================================
// 5. LINEAR LAYER
// ============================================================
struct Linear {
    std::vector<std::vector<ValuePtr>> W;
    std::vector<ValuePtr> b;
    bool use_bias;

    Linear(int n_in, int n_out, bool bias = true)
        : use_bias(bias) {
        double scale = 1.0 / std::sqrt(n_in);
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-scale, scale);
        W.resize(n_out, std::vector<ValuePtr>(n_in));
        for (int i = 0; i < n_out; ++i)
            for (int j = 0; j < n_in; ++j)
                W[i][j] = Value::create(dist(rng));
        if (bias)
            b.resize(n_out, Value::create(0.0));
    }

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) {
        std::vector<ValuePtr> out(W.size());
        for (size_t i = 0; i < W.size(); ++i)
            out[i] = dot(W[i], x);
        if (use_bias)
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = out[i] + b[i];
        return out;
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> p;
        for (auto& row : W) for (auto& w : row) p.push_back(w);
        if (use_bias) for (auto& bi : b) p.push_back(bi);
        return p;
    }
};

// ============================================================
// 6. EMBEDDING
// ============================================================
struct Embedding {
    std::vector<std::vector<ValuePtr>> table;

    Embedding(int vocab_size, int d_model, double scale = 0.02) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-scale, scale);
        table.resize(vocab_size, std::vector<ValuePtr>(d_model));
        for (int i = 0; i < vocab_size; ++i)
            for (int j = 0; j < d_model; ++j)
                table[i][j] = Value::create(dist(rng));
    }

    std::vector<ValuePtr> forward(int idx) {
        std::vector<ValuePtr> out;
        for (auto& v : table[idx])
            out.push_back(v + 0.0); // fresh node
        return out;
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> p;
        for (auto& row : table) for (auto& v : row) p.push_back(v);
        return p;
    }
};

// ============================================================
// 7. POSITIONAL EMBEDDING
// ============================================================
struct PositionalEmbedding {
    std::vector<std::vector<ValuePtr>> table;

    PositionalEmbedding(int max_len, int d_model, double scale = 0.02) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-scale, scale);
        table.resize(max_len, std::vector<ValuePtr>(d_model));
        for (int i = 0; i < max_len; ++i)
            for (int j = 0; j < d_model; ++j)
                table[i][j] = Value::create(dist(rng));
    }

    std::vector<ValuePtr> forward(int pos) {
        std::vector<ValuePtr> out;
        for (auto& v : table[pos])
            out.push_back(v + 0.0);
        return out;
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> p;
        for (auto& row : table) for (auto& v : row) p.push_back(v);
        return p;
    }
};

// ============================================================
// 8. LAYER NORMALIZATION
// ============================================================
struct LayerNorm {
    std::vector<ValuePtr> gamma, beta;
    double eps;

    LayerNorm(int dim, double eps = 1e-5) : eps(eps) {
        for (int i = 0; i < dim; ++i) gamma.push_back(Value::create(1.0));
        for (int i = 0; i < dim; ++i) beta.push_back(Value::create(0.0));
    }

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) {
        size_t n = x.size();
        ValuePtr mean = Value::create(0.0);
        for (auto& xi : x) mean = mean + xi;
        mean = mean / (double)n;
        ValuePtr var = Value::create(0.0);
        for (auto& xi : x) {
            auto diff = xi - mean;
            var = var + diff * diff;
        }
        var = var / (double)n;
        auto std = sqrt(var + eps);
        std::vector<ValuePtr> out(n);
        for (size_t i = 0; i < n; ++i)
            out[i] = gamma[i] * ((x[i] - mean) / std) + beta[i];
        return out;
    }

    std::vector<ValuePtr> parameters() {
        auto p = gamma;
        p.insert(p.end(), beta.begin(), beta.end());
        return p;
    }
};

// ============================================================
// 9. MULTI-HEAD SELF ATTENTION
// ============================================================
struct MultiHeadSelfAttention {
    int d_model, n_heads, d_head;
    Linear Wq, Wk, Wv, Wo;
    Dropout drop;

    MultiHeadSelfAttention(int d_model, int n_heads, double dropout = 0.1)
        : d_model(d_model), n_heads(n_heads), d_head(d_model / n_heads),
          Wq(d_model, d_model, false), Wk(d_model, d_model, false),
          Wv(d_model, d_model, false), Wo(d_model, d_model, false), drop(dropout) {}

    std::vector<ValuePtr> forward(const std::vector<std::vector<ValuePtr>>& X,
                                  const std::vector<int>& pad_mask) {
        int seq_len = X.size();
        std::vector<std::vector<ValuePtr>> Q(seq_len), K(seq_len), V(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            Q[i] = Wq.forward(X[i]);
            K[i] = Wk.forward(X[i]);
            V[i] = Wv.forward(X[i]);
        }
        double scale = 1.0 / std::sqrt(d_head);
        std::vector<std::vector<ValuePtr>> outputs(seq_len, std::vector<ValuePtr>(d_model, Value::create(0.0)));
        for (int h = 0; h < n_heads; ++h) {
            int s = h * d_head, e = s + d_head;
            for (int i = 0; i < seq_len; ++i) {
                std::vector<int> valid_js;
                for (int j = 0; j <= i; ++j) if (pad_mask[j] == 1) valid_js.push_back(j);
                if (valid_js.empty()) valid_js.push_back(i);
                std::vector<ValuePtr> scores;
                for (int j : valid_js) {
                    std::vector<ValuePtr> qh(Q[i].begin()+s, Q[i].begin()+e);
                    std::vector<ValuePtr> kh(K[j].begin()+s, K[j].begin()+e);
                    scores.push_back(dot(qh, kh) * scale);
                }
                auto weights = softmax(scores);
                std::vector<ValuePtr> head_out(d_head, Value::create(0.0));
                for (size_t w_idx = 0; w_idx < valid_js.size(); ++w_idx) {
                    int j = valid_js[w_idx];
                    for (int d = 0; d < d_head; ++d)
                        head_out[d] = head_out[d] + weights[w_idx] * V[j][s + d];
                }
                for (int d = 0; d < d_head; ++d)
                    outputs[i][s + d] = head_out[d];
            }
        }
        std::vector<std::vector<ValuePtr>> out;
        for (int i = 0; i < seq_len; ++i)
            out.push_back(Wo.forward(outputs[i]));
        for (int i = 0; i < seq_len; ++i)
            out[i] = drop.forward(out[i]);
        return out;
    }

    // Incremental with KV cache
    std::vector<ValuePtr> forward_incremental(const std::vector<ValuePtr>& x,
                                             std::vector<std::vector<ValuePtr>>& K_cache,
                                             std::vector<std::vector<ValuePtr>>& V_cache) {
        auto q = Wq.forward(x);
        auto k = Wk.forward(x);
        auto v = Wv.forward(x);
        K_cache.push_back(k);
        V_cache.push_back(v);
        double scale = 1.0 / std::sqrt(d_head);
        std::vector<ValuePtr> out(d_model, Value::create(0.0));
        for (int h = 0; h < n_heads; ++h) {
            int s = h * d_head, e = s + d_head;
            std::vector<ValuePtr> qh(q.begin()+s, q.begin()+e);
            std::vector<ValuePtr> scores;
            for (size_t pos = 0; pos < K_cache.size(); ++pos) {
                std::vector<ValuePtr> kh(K_cache[pos].begin()+s, K_cache[pos].begin()+e);
                scores.push_back(dot(qh, kh) * scale);
            }
            auto weights = softmax(scores);
            std::vector<ValuePtr> head_out(d_head, Value::create(0.0));
            for (size_t pos = 0; pos < V_cache.size(); ++pos) {
                for (int d = 0; d < d_head; ++d)
                    head_out[d] = head_out[d] + weights[pos] * V_cache[pos][s + d];
            }
            for (int d = 0; d < d_head; ++d)
                out[s + d] = head_out[d];
        }
        auto out2 = Wo.forward(out);
        return drop.forward(out2);
    }

    std::vector<ValuePtr> parameters() {
        auto p = Wq.parameters(); auto p2 = Wk.parameters(); auto p3 = Wv.parameters(); auto p4 = Wo.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        p.insert(p.end(), p3.begin(), p3.end());
        p.insert(p.end(), p4.begin(), p4.end());
        return p;
    }
    void set_training(bool mode) { drop.training = mode; }
};

// ============================================================
// 10. FEED-FORWARD
// ============================================================
struct FeedForward {
    Linear fc1, fc2;
    Dropout drop;

    FeedForward(int d_model, int d_ff, double dropout = 0.1)
        : fc1(d_model, d_ff), fc2(d_ff, d_model), drop(dropout) {}

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& x) {
        auto h = fc1.forward(x);
        for (auto& v : h) v = gelu(v);
        h = drop.forward(h);
        return fc2.forward(h);
    }

    std::vector<ValuePtr> parameters() {
        auto p = fc1.parameters();
        auto p2 = fc2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        return p;
    }
    void set_training(bool mode) { drop.training = mode; }
};

// ============================================================
// 11. TRANSFORMER BLOCK
// ============================================================
struct TransformerBlock {
    LayerNorm ln1, ln2;
    MultiHeadSelfAttention attn;
    FeedForward ff;

    TransformerBlock(int d_model, int n_heads, int d_ff, double dropout = 0.1)
        : ln1(d_model), ln2(d_model), attn(d_model, n_heads, dropout), ff(d_model, d_ff, dropout) {}

    std::vector<std::vector<ValuePtr>> forward(const std::vector<std::vector<ValuePtr>>& X,
                                               const std::vector<int>& pad_mask) {
        int seq_len = X.size();
        std::vector<std::vector<ValuePtr>> ln1_out;
        for (auto& x : X) ln1_out.push_back(ln1.forward(x));
        auto attn_out = attn.forward(ln1_out, pad_mask);
        std::vector<std::vector<ValuePtr>> X2(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            X2[i].resize(X[i].size());
            for (size_t j = 0; j < X[i].size(); ++j)
                X2[i][j] = X[i][j] + attn_out[i][j];
        }
        std::vector<std::vector<ValuePtr>> ln2_out;
        for (auto& x : X2) ln2_out.push_back(ln2.forward(x));
        std::vector<std::vector<ValuePtr>> ff_out;
        for (auto& x : ln2_out) ff_out.push_back(ff.forward(x));
        std::vector<std::vector<ValuePtr>> out(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            out[i].resize(X[i].size());
            for (size_t j = 0; j < X[i].size(); ++j)
                out[i][j] = X2[i][j] + ff_out[i][j];
        }
        return out;
    }

    std::vector<ValuePtr> forward_incremental(const std::vector<ValuePtr>& x,
                                             std::vector<std::vector<ValuePtr>>& K_cache,
                                             std::vector<std::vector<ValuePtr>>& V_cache) {
        auto ln1_out = ln1.forward(x);
        auto attn_out = attn.forward_incremental(ln1_out, K_cache, V_cache);
        std::vector<ValuePtr> x2;
        for (size_t i = 0; i < x.size(); ++i) x2.push_back(x[i] + attn_out[i]);
        auto ln2_out = ln2.forward(x2);
        auto ff_out = ff.forward(ln2_out);
        std::vector<ValuePtr> out;
        for (size_t i = 0; i < x2.size(); ++i) out.push_back(x2[i] + ff_out[i]);
        return out;
    }

    std::vector<ValuePtr> parameters() {
        auto p = ln1.parameters();
        auto p2 = attn.parameters();
        auto p3 = ln2.parameters();
        auto p4 = ff.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        p.insert(p.end(), p3.begin(), p3.end());
        p.insert(p.end(), p4.begin(), p4.end());
        return p;
    }
    void set_training(bool mode) { attn.set_training(mode); ff.set_training(mode); }
};

// ============================================================
// 12. MINI-GPT MODEL
// ============================================================
struct MiniGPT {
    int d_model, max_len, vocab_size;
    Embedding embed;
    PositionalEmbedding pos_embed;
    Dropout embed_drop;
    std::vector<TransformerBlock> blocks;
    LayerNorm ln_f;
    Linear head;
    std::vector<std::pair<std::vector<std::vector<ValuePtr>>, std::vector<std::vector<ValuePtr>>>> caches; // per block: (K,V)

    MiniGPT(int vocab_size, int d_model = 16, int n_heads = 2, int n_layers = 2, int d_ff = 32,
            int max_len = 64, double dropout = 0.1)
        : d_model(d_model), max_len(max_len), vocab_size(vocab_size),
          embed(vocab_size, d_model), pos_embed(max_len, d_model), embed_drop(dropout),
          ln_f(d_model), head(d_model, vocab_size) {
        for (int i = 0; i < n_layers; ++i)
            blocks.emplace_back(d_model, n_heads, d_ff, dropout);
    }

    std::vector<std::vector<ValuePtr>> forward(const std::vector<int>& token_ids,
                                               const std::vector<int>& pad_mask = {}) {
        std::vector<std::vector<ValuePtr>> X;
        for (size_t pos = 0; pos < token_ids.size(); ++pos) {
            auto emb = embed.forward(token_ids[pos]);
            auto pe = pos_embed.forward(pos);
            std::vector<ValuePtr> combined;
            for (int d = 0; d < d_model; ++d)
                combined.push_back(emb[d] + pe[d]);
            X.push_back(combined);
        }
        for (auto& x : X) x = embed_drop.forward(x);
        for (auto& block : blocks)
            X = block.forward(X, pad_mask);
        std::vector<std::vector<ValuePtr>> ln_out;
        for (auto& x : X) ln_out.push_back(ln_f.forward(x));
        std::vector<std::vector<ValuePtr>> logits_seq;
        for (auto& x : ln_out) logits_seq.push_back(head.forward(x));
        return logits_seq;
    }

    void init_cache() {
        caches.clear();
        for (size_t i = 0; i < blocks.size(); ++i)
            caches.push_back({{}, {}});
    }

    std::vector<ValuePtr> forward_incremental(int token_id, int pos) {
        if (pos >= max_len) throw std::runtime_error("pos exceeds max_len");
        auto emb = embed.forward(token_id);
        auto pe = pos_embed.forward(pos);
        std::vector<ValuePtr> x;
        for (int d = 0; d < d_model; ++d) x.push_back(emb[d] + pe[d]);
        x = embed_drop.forward(x);
        for (size_t i = 0; i < blocks.size(); ++i) {
            x = blocks[i].forward_incremental(x, caches[i].first, caches[i].second);
        }
        auto ln_out = ln_f.forward(x);
        return head.forward(ln_out);
    }

    std::vector<ValuePtr> parameters() {
        auto p = embed.parameters();
        auto p2 = pos_embed.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        for (auto& b : blocks) {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto p_ln = ln_f.parameters();
        p.insert(p.end(), p_ln.begin(), p_ln.end());
        auto p_head = head.parameters();
        p.insert(p.end(), p_head.begin(), p_head.end());
        return p;
    }
    void set_training(bool mode) {
        embed_drop.training = mode;
        for (auto& b : blocks) b.set_training(mode);
    }
};

// ============================================================
// 13. LOSS FUNCTION
// ============================================================
ValuePtr cross_entropy_loss(const std::vector<std::vector<ValuePtr>>& logits_seq,
                            const std::vector<int>& target_ids,
                            const std::vector<int>& pad_mask) {
    std::vector<ValuePtr> losses;
    for (size_t i = 0; i < logits_seq.size(); ++i) {
        if (pad_mask[i] == 0) continue;
        auto probs = softmax(logits_seq[i]);
        losses.push_back(log(probs[target_ids[i]]) * -1.0);
    }
    if (losses.empty()) return Value::create(0.0);
    ValuePtr total = Value::create(0.0);
    for (auto& l : losses) total = total + l;
    return total / (double)losses.size();
}

// ============================================================
// 14. OPTIMIZER: ADAMW
// ============================================================
struct AdamW {
    std::vector<ValuePtr> params;
    double lr, b1, b2, eps, wd;
    std::vector<double> m, v;
    int t;

    AdamW(std::vector<ValuePtr> params, double lr = 0.01, double betas1 = 0.9, double betas2 = 0.999,
          double eps = 1e-8, double weight_decay = 0.01)
        : params(params), lr(lr), b1(betas1), b2(betas2), eps(eps), wd(weight_decay),
          m(params.size(), 0.0), v(params.size(), 0.0), t(0) {}

    void step() {
        t++;
        double bc1 = 1 - std::pow(b1, t);
        double bc2 = 1 - std::pow(b2, t);
        for (size_t i = 0; i < params.size(); ++i) {
            double g = params[i]->grad;
            m[i] = b1 * m[i] + (1 - b1) * g;
            v[i] = b2 * v[i] + (1 - b2) * g * g;
            double m_hat = m[i] / bc1;
            double v_hat = v[i] / bc2;
            params[i]->data -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * params[i]->data);
        }
    }

    void zero_grad() {
        for (auto& p : params) p->grad = 0.0;
    }
};

// ============================================================
// 15. GRADIENT CLIPPING
// ============================================================
void clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    double total_sq = 0.0;
    for (auto& p : params) total_sq += p->grad * p->grad;
    double norm = std::sqrt(total_sq);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (auto& p : params) p->grad *= scale;
    }
}

// ============================================================
// 16. SCHEDULER: WARMUP + COSINE
// ============================================================
struct WarmupCosineScheduler {
    AdamW* opt;
    int warmup, total;
    double base_lr, min_lr;
    int step_num;

    WarmupCosineScheduler(AdamW* opt, int warmup_steps, int total_steps, double base_lr, double min_lr = 1e-5)
        : opt(opt), warmup(warmup_steps), total(total_steps), base_lr(base_lr), min_lr(min_lr), step_num(0) {}

    double step() {
        step_num++;
        double lr;
        if (step_num < warmup)
            lr = base_lr * step_num / warmup;
        else {
            double progress = (double)(step_num - warmup) / (total - warmup);
            progress = std::min(1.0, progress);
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + std::cos(M_PI * progress));
        }
        opt->lr = lr;
        return lr;
    }
};

// ============================================================
// 17. GENERATION HELPER
// ============================================================
int sample_from_logits(const std::vector<ValuePtr>& logits, double temperature, int top_k, double top_p) {
    std::vector<double> scaled;
    for (auto& v : logits) scaled.push_back(v->data / std::max(temperature, 1e-6));
    double m = *std::max_element(scaled.begin(), scaled.end());
    std::vector<double> exps;
    for (auto& s : scaled) exps.push_back(std::exp(s - m));
    double sum_e = std::accumulate(exps.begin(), exps.end(), 0.0);
    std::vector<double> probs(exps.size());
    for (size_t i = 0; i < exps.size(); ++i) probs[i] = exps[i] / sum_e;

    if (top_k > 0) {
        std::vector<size_t> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        for (size_t i = top_k; i < idx.size(); ++i) probs[idx[i]] = 0.0;
    }
    if (top_p > 0.0) {
        std::vector<size_t> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        double cum = 0.0;
        for (size_t i = 0; i < idx.size(); ++i) {
            cum += probs[idx[i]];
            if (cum >= top_p) {
                for (size_t j = i+1; j < idx.size(); ++j) probs[idx[j]] = 0.0;
                break;
            }
        }
    }
    double sum_p = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (sum_p <= 0) return std::max_element(scaled.begin(), scaled.end()) - scaled.begin();
    for (auto& p : probs) p /= sum_p;
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return probs.size() - 1;
}

std::string generate(MiniGPT& model, ByteLevelBPETokenizer& tok, const std::string& prompt,
                     int max_new_tokens = 25, double temperature = 0.9, int top_k = 0, double top_p = 0.9) {
    model.set_training(false);
    model.init_cache();
    auto ids = tok.encode(prompt, true, false);
    int eos_id = tok.vocab["<eos>"];
    std::vector<ValuePtr> logits;
    for (size_t pos = 0; pos < ids.size(); ++pos)
        logits = model.forward_incremental(ids[pos], pos);
    for (int _ = 0; _ < max_new_tokens; ++_) {
        int next_id = sample_from_logits(logits, temperature, top_k, top_p);
        ids.push_back(next_id);
        if (next_id == eos_id) break;
        int pos = ids.size() - 1;
        if (pos >= model.max_len) break;
        logits = model.forward_incremental(next_id, pos);
    }
    model.set_training(true);
    return tok.decode(ids);
}

// ============================================================
// 18. PYTHON BINDINGS
// ============================================================
namespace pybind11 { namespace detail {
    // Hash untuk pair<string,string> agar bisa dipakai di unordered_map
    template <> struct type_caster<std::pair<std::string, std::string>> {
    public:
        using pair_type = std::pair<std::string, std::string>;
        PYBIND11_TYPE_CASTER(pair_type, _("Tuple[str, str]"));
        bool load(handle src, bool convert) {
            if (!py::isinstance<py::tuple>(src)) return false;
            auto t = py::cast<py::tuple>(src);
            if (t.size() != 2) return false;
            value.first = t[0].cast<std::string>();
            value.second = t[1].cast<std::string>();
            return true;
        }
        static handle cast(pair_type src, return_value_policy policy, handle parent) {
            py::tuple t(2);
            t[0] = py::cast(src.first);
            t[1] = py::cast(src.second);
            return t.release();
        }
    };
}}

PYBIND11_MODULE(minigpt, m) {
    m.doc() = "Mini-GPT C++ implementation (optimized)";

    // Value
    py::class_<Value, ValuePtr>(m, "Value")
        .def(py::init<double>())
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def("backward", &Value::backward)
        .def("__repr__", &Value::repr)
        .def("__add__", [](ValuePtr a, ValuePtr b) { return a + b; })
        .def("__add__", [](ValuePtr a, double b) { return a + b; })
        .def("__radd__", [](ValuePtr a, double b) { return b + a; })
        .def("__mul__", [](ValuePtr a, ValuePtr b) { return a * b; })
        .def("__mul__", [](ValuePtr a, double b) { return a * b; })
        .def("__rmul__", [](ValuePtr a, double b) { return b * a; })
        .def("__sub__", [](ValuePtr a, ValuePtr b) { return a - b; })
        .def("__sub__", [](ValuePtr a, double b) { return a - b; })
        .def("__rsub__", [](ValuePtr a, double b) { return b - a; })
        .def("__truediv__", [](ValuePtr a, ValuePtr b) { return a / b; })
        .def("__truediv__", [](ValuePtr a, double b) { return a / b; })
        .def("__rtruediv__", [](ValuePtr a, double b) { return b / a; })
        .def("__pow__", [](ValuePtr a, double exp) { return pow(a, exp); })
        .def("exp", [](ValuePtr a) { return exp(a); })
        .def("log", [](ValuePtr a) { return log(a); })
        .def("sqrt", [](ValuePtr a) { return sqrt(a); })
        .def("tanh", [](ValuePtr a) { return tanh(a); })
        .def("relu", [](ValuePtr a) { return relu(a); })
        .def("gelu", [](ValuePtr a) { return gelu(a); });

    // Tokenizer
    py::class_<ByteLevelBPETokenizer>(m, "ByteLevelBPETokenizer")
        .def(py::init<>())
        .def("train", &ByteLevelBPETokenizer::train,
             py::arg("corpus"), py::arg("vocab_size") = 400)
        .def("encode", &ByteLevelBPETokenizer::encode,
             py::arg("text"), py::arg("add_bos") = false, py::arg("add_eos") = false)
        .def("decode", &ByteLevelBPETokenizer::decode)
        .def("save", &ByteLevelBPETokenizer::save)
        .def("load", &ByteLevelBPETokenizer::load)
        .def_readwrite("vocab", &ByteLevelBPETokenizer::vocab)
        .def_readwrite("inv_vocab", &ByteLevelBPETokenizer::inv_vocab)
        .def_readwrite("merge_order", &ByteLevelBPETokenizer::merge_order);

    // Layers and model
    py::class_<Dropout>(m, "Dropout")
        .def(py::init<double>(), py::arg("p") = 0.1)
        .def("forward", &Dropout::forward)
        .def_readwrite("training", &Dropout::training);

    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, bool>(),
             py::arg("n_in"), py::arg("n_out"), py::arg("bias") = true)
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);

    py::class_<Embedding>(m, "Embedding")
        .def(py::init<int, int, double>(),
             py::arg("vocab_size"), py::arg("d_model"), py::arg("scale") = 0.02)
        .def("forward", &Embedding::forward)
        .def("parameters", &Embedding::parameters);

    py::class_<PositionalEmbedding>(m, "PositionalEmbedding")
        .def(py::init<int, int, double>(),
             py::arg("max_len"), py::arg("d_model"), py::arg("scale") = 0.02)
        .def("forward", &PositionalEmbedding::forward)
        .def("parameters", &PositionalEmbedding::parameters);

    py::class_<LayerNorm>(m, "LayerNorm")
        .def(py::init<int, double>(),
             py::arg("dim"), py::arg("eps") = 1e-5)
        .def("forward", &LayerNorm::forward)
        .def("parameters", &LayerNorm::parameters);

    py::class_<MultiHeadSelfAttention>(m, "MultiHeadSelfAttention")
        .def(py::init<int, int, double>(),
             py::arg("d_model"), py::arg("n_heads"), py::arg("dropout") = 0.1)
        .def("forward", &MultiHeadSelfAttention::forward)
        .def("forward_incremental", &MultiHeadSelfAttention::forward_incremental)
        .def("parameters", &MultiHeadSelfAttention::parameters)
        .def("set_training", &MultiHeadSelfAttention::set_training);

    py::class_<FeedForward>(m, "FeedForward")
        .def(py::init<int, int, double>(),
             py::arg("d_model"), py::arg("d_ff"), py::arg("dropout") = 0.1)
        .def("forward", &FeedForward::forward)
        .def("parameters", &FeedForward::parameters)
        .def("set_training", &FeedForward::set_training);

    py::class_<TransformerBlock>(m, "TransformerBlock")
        .def(py::init<int, int, int, double>(),
             py::arg("d_model"), py::arg("n_heads"), py::arg("d_ff"), py::arg("dropout") = 0.1)
        .def("forward", &TransformerBlock::forward)
        .def("forward_incremental", &TransformerBlock::forward_incremental)
        .def("parameters", &TransformerBlock::parameters)
        .def("set_training", &TransformerBlock::set_training);

    py::class_<MiniGPT>(m, "MiniGPT")
        .def(py::init<int, int, int, int, int, int, double>(),
             py::arg("vocab_size"), py::arg("d_model") = 16, py::arg("n_heads") = 2,
             py::arg("n_layers") = 2, py::arg("d_ff") = 32, py::arg("max_len") = 64,
             py::arg("dropout") = 0.1)
        .def("forward", &MiniGPT::forward,
             py::arg("token_ids"), py::arg("pad_mask") = std::vector<int>())
        .def("init_cache", &MiniGPT::init_cache)
        .def("forward_incremental", &MiniGPT::forward_incremental)
        .def("parameters", &MiniGPT::parameters)
        .def("set_training", &MiniGPT::set_training);

    // Optimizer & Scheduler
    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<ValuePtr>, double, double, double, double, double>(),
             py::arg("params"), py::arg("lr") = 0.01, py::arg("betas1") = 0.9,
             py::arg("betas2") = 0.999, py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.01)
        .def("step", &AdamW::step)
        .def("zero_grad", &AdamW::zero_grad)
        .def_readwrite("lr", &AdamW::lr);

    py::class_<WarmupCosineScheduler>(m, "WarmupCosineScheduler")
        .def(py::init<AdamW*, int, int, double, double>(),
             py::arg("optimizer"), py::arg("warmup_steps"), py::arg("total_steps"),
             py::arg("base_lr"), py::arg("min_lr") = 1e-5)
        .def("step", &WarmupCosineScheduler::step);

    // Utility functions
    m.def("clip_grad_norm", &clip_grad_norm);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("sample_from_logits", &sample_from_logits);
    m.def("generate", &generate,
          py::arg("model"), py::arg("tokenizer"), py::arg("prompt"),
          py::arg("max_new_tokens") = 25, py::arg("temperature") = 0.9,
          py::arg("top_k") = 0, py::arg("top_p") = 0.9);
}