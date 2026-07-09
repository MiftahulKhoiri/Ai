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
#include <queue>
#include <limits>
#include <cassert>

namespace py = pybind11;

// ============================================================
// 1. VALUE (Autograd Engine)
// ============================================================
struct Value : std::enable_shared_from_this<Value> {
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Value>> _prev;
    std::string _op;

    Value(double data, const std::vector<std::shared_ptr<Value>>& children = {}, std::string op = "")
        : data(data), grad(0.0), _backward([]{}), _prev(children), _op(op) {}

    static std::shared_ptr<Value> create(double data) {
        return std::make_shared<Value>(data);
    }

    // Operator overloading returning shared_ptr
    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other) {
        auto out = Value::create(data + other->data);
        out->_prev = {shared_from_this(), other};
        out->_op = "+";
        auto self = shared_from_this();
        out->_backward = [self, other, out]() {
            self->grad += out->grad;
            other->grad += out->grad;
        };
        return out;
    }
    std::shared_ptr<Value> operator+(double other) { return *this + Value::create(other); }

    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other) {
        auto out = Value::create(data * other->data);
        out->_prev = {shared_from_this(), other};
        out->_op = "*";
        auto self = shared_from_this();
        out->_backward = [self, other, out]() {
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
        };
        return out;
    }
    std::shared_ptr<Value> operator*(double other) { return *this * Value::create(other); }

    std::shared_ptr<Value> pow(double exponent) {
        auto out = Value::create(std::pow(data, exponent));
        out->_prev = {shared_from_this()};
        out->_op = "**" + std::to_string(exponent);
        auto self = shared_from_this();
        out->_backward = [self, exponent, out]() {
            self->grad += exponent * std::pow(self->data, exponent - 1) * out->grad;
        };
        return out;
    }

    std::shared_ptr<Value> exp() {
        double x = std::max(std::min(data, 60.0), -60.0);
        auto out = Value::create(std::exp(x));
        out->_prev = {shared_from_this()};
        out->_op = "exp";
        auto self = shared_from_this();
        out->_backward = [self, out]() { self->grad += out->data * out->grad; };
        return out;
    }

    std::shared_ptr<Value> log() {
        double x = std::max(data, 1e-12);
        auto out = Value::create(std::log(x));
        out->_prev = {shared_from_this()};
        out->_op = "log";
        auto self = shared_from_this();
        out->_backward = [self, x, out]() { self->grad += (1.0 / x) * out->grad; };
        return out;
    }

    std::shared_ptr<Value> sqrt() {
        double x = std::max(data, 1e-12);
        double r = std::sqrt(x);
        auto out = Value::create(r);
        out->_prev = {shared_from_this()};
        out->_op = "sqrt";
        auto self = shared_from_this();
        out->_backward = [self, r, out]() { self->grad += (0.5 / r) * out->grad; };
        return out;
    }

    std::shared_ptr<Value> tanh() {
        double t = std::tanh(data);
        auto out = Value::create(t);
        out->_prev = {shared_from_this()};
        out->_op = "tanh";
        auto self = shared_from_this();
        out->_backward = [self, t, out]() { self->grad += (1 - t * t) * out->grad; };
        return out;
    }

    std::shared_ptr<Value> relu() {
        double out_data = data > 0 ? data : 0.0;
        auto out = Value::create(out_data);
        out->_prev = {shared_from_this()};
        out->_op = "relu";
        auto self = shared_from_this();
        out->_backward = [self, out]() { if (self->data > 0) self->grad += out->grad; };
        return out;
    }

    std::shared_ptr<Value> gelu() {
        double c = 0.7978845608028654;  // sqrt(2/pi)
        auto inner = (*this + (*this).pow(3) * 0.044715) * c;
        return (*this * (inner->tanh() + 1.0)) * 0.5;
    }

    // Negation and subtraction
    std::shared_ptr<Value> operator-() { return *this * -1; }
    std::shared_ptr<Value> operator-(std::shared_ptr<Value> other) { return *this + (-*other); }
    std::shared_ptr<Value> operator-(double other) { return *this - Value::create(other); }
    std::shared_ptr<Value> operator/(std::shared_ptr<Value> other) {
        return *this * other->pow(-1);
    }
    std::shared_ptr<Value> operator/(double other) { return *this / Value::create(other); }

    // Reverse operators (for double on left)
    friend std::shared_ptr<Value> operator+(double lhs, std::shared_ptr<Value> rhs) { return Value::create(lhs) + rhs; }
    friend std::shared_ptr<Value> operator*(double lhs, std::shared_ptr<Value> rhs) { return Value::create(lhs) * rhs; }
    friend std::shared_ptr<Value> operator-(double lhs, std::shared_ptr<Value> rhs) { return Value::create(lhs) - rhs; }
    friend std::shared_ptr<Value> operator/(double lhs, std::shared_ptr<Value> rhs) { return Value::create(lhs) / rhs; }

    void backward() {
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<Value*> visited;
        std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
            if (visited.count(v.get())) return;
            visited.insert(v.get());
            for (auto& child : v->_prev) build_topo(child);
            topo.push_back(v);
        };
        build_topo(shared_from_this());
        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Value(data=" << data << ", grad=" << grad << ")";
        return oss.str();
    }
};

// ============================================================
// 2. TOKENIZER BYTE-LEVEL BPE
// ============================================================
static std::unordered_map<int, std::string> _bytes_to_unicode() {
    std::unordered_map<int, std::string> b2u;
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xa1; b <= 0xac; ++b) bs.push_back(b);
    for (int b = 0xae; b <= 0xff; ++b) bs.push_back(b);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            b2u[b] = std::string(1, static_cast<char>(256 + n));
            ++n;
        } else {
            b2u[b] = std::string(1, static_cast<char>(b));
        }
    }
    return b2u;
}

static std::unordered_map<int, std::string> BYTE_ENCODER = _bytes_to_unicode();
static std::unordered_map<std::string, int> BYTE_DECODER;
static bool byte_decoder_initialized = false;
void init_byte_decoder() {
    if (!byte_decoder_initialized) {
        for (auto& p : BYTE_ENCODER) BYTE_DECODER[p.second] = p.first;
        byte_decoder_initialized = true;
    }
}

static const std::regex PRETOKEN_PAT(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)"
);

struct ByteLevelBPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::unordered_map<std::pair<std::string, std::string>, std::string, py::detail::pair_hasher> merges;
    std::vector<std::pair<std::string, std::string>> merge_order;
    std::unordered_map<std::pair<std::string, std::string>, int, py::detail::pair_hasher> merge_rank;

    void train(const std::string& corpus, int vocab_size = 400) {
        // Tokenization into byte-level symbols
        std::vector<std::vector<std::string>> words;
        std::sregex_iterator it(corpus.begin(), corpus.end(), PRETOKEN_PAT);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            std::string token = it->str();
            std::vector<std::string> syms;
            for (unsigned char c : token) {
                syms.push_back(BYTE_ENCODER.at(c));
            }
            words.push_back(syms);
        }

        std::map<std::vector<std::string>, int> freq;
        for (auto& w : words) freq[w]++;

        std::map<std::vector<std::string>, std::vector<std::string>> splits;
        for (auto& p : freq) splits[p.first] = p.first;

        std::set<std::string> base_vocab;
        for (auto& p : BYTE_ENCODER) base_vocab.insert(p.second);

        int num_merges = std::max(0, vocab_size - (int)base_vocab.size() - 4); // special tokens
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
            merges[best_pair] = merged;
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

        // Build vocab
        std::vector<std::string> specials = {"<pad>", "<bos>", "<eos>", "<unk>"};
        std::vector<std::string> all_tokens = specials;
        for (auto& s : base_vocab) all_tokens.push_back(s);
        // Add merged tokens
        for (auto& p : merges) all_tokens.push_back(p.second);
        std::sort(all_tokens.begin() + 4, all_tokens.end());
        all_tokens.erase(std::unique(all_tokens.begin() + 4, all_tokens.end()), all_tokens.end());
        for (size_t i = 0; i < all_tokens.size(); ++i) {
            vocab[all_tokens[i]] = i;
            inv_vocab[i] = all_tokens[i];
        }
        for (size_t i = 0; i < merge_order.size(); ++i) {
            merge_rank[merge_order[i]] = i;
        }
    }

    std::vector<std::string> _apply_bpe(const std::vector<std::string>& symbols) {
        std::vector<std::string> word = symbols;
        while (word.size() > 1) {
            std::pair<std::string, std::string> best_pair;
            int best_rank = std::numeric_limits<int>::max();
            bool found = false;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                std::pair<std::string, std::string> p = {word[i], word[i+1]};
                auto it = merge_rank.find(p);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_pair = p;
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

    std::vector<int> encode(const std::string& text, bool add_bos=false, bool add_eos=false) {
        std::vector<int> ids;
        if (add_bos) ids.push_back(vocab["<bos>"]);
        std::sregex_iterator it(text.begin(), text.end(), PRETOKEN_PAT);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            std::string token = it->str();
            std::vector<std::string> syms;
            for (unsigned char c : token) syms.push_back(BYTE_ENCODER.at(c));
            for (auto& s : _apply_bpe(syms)) {
                auto vit = vocab.find(s);
                ids.push_back(vit != vocab.end() ? vit->second : vocab["<unk>"]);
            }
        }
        if (add_eos) ids.push_back(vocab["<eos>"]);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string chars;
        for (int id : ids) {
            auto it = inv_vocab.find(id);
            if (it != inv_vocab.end()) {
                std::string tok = it->second;
                if (tok == "<pad>" || tok == "<bos>" || tok == "<eos>" || tok == "<unk>")
                    continue;
                chars += tok;
            }
        }
        std::string bytes;
        for (size_t i = 0; i < chars.size(); ) {
            // Karena beberapa karakter unicode bisa terdiri dari beberapa byte, tetapi di sini karakter disimpan sebagai char (1 byte)
            // Kita asumsikan BYTE_DECODER memetakan karakter unicode kembali ke byte tunggal.
            // Ini hanya bekerja jika representasi string cocok.
            char c = chars[i];
            std::string s(1, c);
            if (BYTE_DECODER.count(s)) {
                bytes.push_back(static_cast<char>(BYTE_DECODER[s]));
            }
            i++;
        }
        return bytes; // as UTF-8
    }

    void save(const std::string& path) {
        py::dict data;
        data["vocab"] = vocab;
        std::vector<py::tuple> mo;
        for (auto& p : merge_order) mo.push_back(py::make_tuple(p.first, p.second));
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
        std::vector<py::tuple> mo = data["merge_order"].cast<std::vector<py::tuple>>();
        merge_order.clear();
        merges.clear();
        merge_rank.clear();
        for (auto& t : mo) {
            std::string a = t[0].cast<std::string>();
            std::string b = t[1].cast<std::string>();
            merge_order.push_back({a, b});
            merges[{a, b}] = a + b;
        }
        for (size_t i = 0; i < merge_order.size(); ++i)
            merge_rank[merge_order[i]] = i;
    }
};

// ============================================================
// 3. UTILITAS VEKTOR / MATRIKS (opsional, karena kita akan gunakan ValuePtr)
// ============================================================
using ValuePtr = std::shared_ptr<Value>;
using Vec = std::vector<ValuePtr>;
using Mat = std::vector<Vec>;

Vec make_vector(int n, double scale = 0.1) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-scale, scale);
    Vec v(n);
    for (int i = 0; i < n; ++i) v[i] = Value::create(dist(rng));
    return v;
}

Mat make_matrix(int rows, int cols, double scale = 0.1) {
    Mat m(rows);
    for (int i = 0; i < rows; ++i) m[i] = make_vector(cols, scale);
    return m;
}

ValuePtr dot(const Vec& a, const Vec& b) {
    ValuePtr s = Value::create(0.0);
    for (size_t i = 0; i < a.size(); ++i) s = *s + *a[i] * b[i];
    return s;
}

Vec softmax(const Vec& logits) {
    double m = std::max_element(logits.begin(), logits.end(),
        [](ValuePtr a, ValuePtr b) { return a->data < b->data; })->get()->data;
    Vec exps;
    for (auto& x : logits) exps.push_back((*x - m)->exp());
    ValuePtr total = Value::create(0.0);
    for (auto& e : exps) total = *total + *e;
    Vec probs;
    for (auto& e : exps) probs.push_back(*e / total);
    return probs;
}

// ============================================================
// 4. DROPOUT
// ============================================================
struct Dropout {
    double p;
    bool training;
    std::mt19937 rng;

    Dropout(double p = 0.1) : p(p), training(true), rng(42) {}

    Vec forward(const Vec& x) {
        if (!training || p <= 0) return x;
        double keep = 1.0 - p;
        std::uniform_real_distribution<double> dist(0, 1);
        Vec out;
        for (auto& v : x) {
            if (dist(rng) < keep)
                out.push_back(*v / keep);
            else
                out.push_back(*v * 0.0);
        }
        return out;
    }
};

// ============================================================
// 5. BLOK ARSITEKTUR TRANSFORMER
// ============================================================

// Helper: parameter collector base class
struct Module {
    virtual std::vector<ValuePtr> parameters() = 0;
    virtual void train() {}
    virtual void eval() {}
    virtual ~Module() = default;
};

struct Linear : Module {
    Mat W;
    Vec b;
    bool use_bias;

    Linear(int n_in, int n_out, bool bias = true)
        : W(make_matrix(n_out, n_in, 1.0/std::sqrt(n_in))), use_bias(bias) {
        if (use_bias) b = make_vector(n_out, 0.0);
    }

    Vec forward(const Vec& x) {
        Vec out(W.size());
        for (size_t i = 0; i < W.size(); ++i) out[i] = dot(W[i], x);
        if (use_bias) {
            for (size_t i = 0; i < out.size(); ++i) out[i] = *out[i] + b[i];
        }
        return out;
    }

    std::vector<ValuePtr> parameters() override {
        std::vector<ValuePtr> p;
        for (auto& row : W) for (auto& w : row) p.push_back(w);
        if (use_bias) for (auto& bi : b) p.push_back(bi);
        return p;
    }
};

struct Embedding : Module {
    Mat table;

    Embedding(int vocab_size, int d_model, double scale = 0.02) {
        table = make_matrix(vocab_size, d_model, scale);
    }

    Vec forward(int idx) {
        Vec out = table[idx];
        // Identity op (fresh nodes)
        Vec res;
        for (auto& v : out) res.push_back(*v + 0.0);
        return res;
    }

    std::vector<ValuePtr> parameters() override {
        std::vector<ValuePtr> p;
        for (auto& row : table) for (auto& v : row) p.push_back(v);
        return p;
    }
};

struct PositionalEmbedding : Module {
    Mat table;

    PositionalEmbedding(int max_len, int d_model, double scale = 0.02) {
        table = make_matrix(max_len, d_model, scale);
    }

    Vec forward(int pos) {
        Vec out = table[pos];
        Vec res;
        for (auto& v : out) res.push_back(*v + 0.0);
        return res;
    }

    std::vector<ValuePtr> parameters() override {
        std::vector<ValuePtr> p;
        for (auto& row : table) for (auto& v : row) p.push_back(v);
        return p;
    }
};

struct LayerNorm : Module {
    Vec gamma, beta;
    double eps;

    LayerNorm(int dim, double eps = 1e-5) : gamma(make_vector(dim, 1.0)), beta(make_vector(dim, 0.0)), eps(eps) {}

    Vec forward(const Vec& x) {
        int n = x.size();
        ValuePtr mean = Value::create(0.0);
        for (auto& xi : x) mean = *mean + *xi;
        mean = *mean / n;
        ValuePtr var = Value::create(0.0);
        for (auto& xi : x) {
            auto diff = *xi - mean;
            var = *var + *diff * diff;
        }
        var = *var / n;
        auto std = (var + eps)->sqrt();
        Vec out(n);
        for (int i = 0; i < n; ++i) {
            out[i] = *(*x[i] - mean) / std;
            out[i] = *out[i] * gamma[i] + beta[i];
        }
        return out;
    }

    std::vector<ValuePtr> parameters() override {
        std::vector<ValuePtr> p = gamma;
        p.insert(p.end(), beta.begin(), beta.end());
        return p;
    }
};

struct MultiHeadSelfAttention : Module {
    int d_model, n_heads, d_head;
    Linear Wq, Wk, Wv, Wo;
    Dropout drop;

    MultiHeadSelfAttention(int d_model, int n_heads, double dropout = 0.1)
        : d_model(d_model), n_heads(n_heads), d_head(d_model/n_heads),
          Wq(d_model, d_model, false), Wk(d_model, d_model, false), Wv(d_model, d_model, false),
          Wo(d_model, d_model, false), drop(dropout) {}

    Vec forward(const Vec& X, const std::vector<int>& pad_mask) { // X: seq_len vectors
        int seq_len = X.size();
        Vec Q(seq_len), K(seq_len), V(seq_len);
        for (int i = 0; i < seq_len; ++i) { Q[i] = Wq.forward({X[i]})[0]; K[i] = Wk.forward({X[i]})[0]; V[i] = Wv.forward({X[i]})[0]; } // Naive: linear on single vector

        // Real implementation would process batch, but for simplicity we do per-token linear
        // (the above is wrong, needs proper reshaping; we'll use explicit loops with vector operations)
        // Let's redo: Q = Wq.forward(X) as list of vectors
        // Implement a helper that applies Linear to each vector in list
        auto apply_linear = [](Linear& lin, const Vec& X) {
            Vec out;
            for (auto& x : X) out.push_back(lin.forward(x)[0]);
            return out;
        };
        Vec Q_ = apply_linear(Wq, X);
        Vec K_ = apply_linear(Wk, X);
        Vec V_ = apply_linear(Wv, X);

        double scale = 1.0 / std::sqrt(d_head);
        Vec outputs(seq_len);
        for (int i = 0; i < seq_len; ++i) outputs[i] = Vec(d_model); // initialize zeros

        for (int h = 0; h < n_heads; ++h) {
            int s = h * d_head, e = s + d_head;
            for (int i = 0; i < seq_len; ++i) {
                std::vector<int> valid_js;
                for (int j = 0; j <= i; ++j) if (pad_mask[j] == 1) valid_js.push_back(j);
                if (valid_js.empty()) valid_js.push_back(i);
                Vec scores;
                for (int j : valid_js) {
                    Vec qh_i(Q_[i].begin()+s, Q_[i].begin()+e);
                    Vec kh_j(K_[j].begin()+s, K_[j].begin()+e);
                    scores.push_back(*dot(qh_i, kh_j) * scale);
                }
                Vec weights = softmax(scores);
                Vec head_out(d_head, Value::create(0.0));
                for (size_t w_idx = 0; w_idx < valid_js.size(); ++w_idx) {
                    int j = valid_js[w_idx];
                    Vec vh_j(V_[j].begin()+s, V_[j].begin()+e);
                    for (int d = 0; d < d_head; ++d) {
                        head_out[d] = *head_out[d] + *weights[w_idx] * vh_j[d];
                    }
                }
                for (int d = 0; d < d_head; ++d) outputs[i][s + d] = head_out[d];
            }
        }

        Vec outs = apply_linear(Wo, outputs);
        outs = drop.forward(outs);
        return outs;
    }

    // Incremental with KV cache
    Vec forward_incremental(const Vec& x, std::vector<Vec>& K_cache, std::vector<Vec>& V_cache) {
        // x is single vector (d_model)
        Vec q = Wq.forward({x})[0];
        Vec k = Wk.forward({x})[0];
        Vec v = Wv.forward({x})[0];
        K_cache.push_back(k);
        V_cache.push_back(v);
        double scale = 1.0 / std::sqrt(d_head);
        Vec out(d_model, Value::create(0.0));
        for (int h = 0; h < n_heads; ++h) {
            int s = h * d_head, e = s + d_head;
            Vec qh(q.begin()+s, q.begin()+e);
            Vec scores;
            for (size_t pos = 0; pos < K_cache.size(); ++pos) {
                Vec kh(K_cache[pos].begin()+s, K_cache[pos].begin()+e);
                scores.push_back(*dot(qh, kh) * scale);
            }
            Vec weights = softmax(scores);
            Vec head_out(d_head, Value::create(0.0));
            for (size_t pos = 0; pos < V_cache.size(); ++pos) {
                Vec vh(V_cache[pos].begin()+s, V_cache[pos].begin()+e);
                for (int d = 0; d < d_head; ++d) head_out[d] = *head_out[d] + *weights[pos] * vh[d];
            }
            for (int d = 0; d < d_head; ++d) out[s + d] = head_out[d];
        }
        Vec out2 = Wo.forward(out);
        return drop.forward(out2);
    }

    std::vector<ValuePtr> parameters() override {
        auto p = Wq.parameters(); auto p2 = Wk.parameters(); auto p3 = Wv.parameters(); auto p4 = Wo.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        p.insert(p.end(), p3.begin(), p3.end());
        p.insert(p.end(), p4.begin(), p4.end());
        return p;
    }
    void train() override { drop.training = true; }
    void eval() override { drop.training = false; }
};

struct FeedForward : Module {
    Linear fc1, fc2;
    Dropout drop;

    FeedForward(int d_model, int d_ff, double dropout = 0.1)
        : fc1(d_model, d_ff), fc2(d_ff, d_model), drop(dropout) {}

    Vec forward(const Vec& x) {
        Vec h;
        for (auto& xi : x) h.push_back(fc1.forward({xi})[0]); // not efficient but works
        // apply gelu
        for (size_t i = 0; i < h.size(); ++i) h[i] = h[i]->gelu();
        h = drop.forward(h);
        Vec out;
        for (auto& hi : h) out.push_back(fc2.forward({hi})[0]);
        return out;
    }

    std::vector<ValuePtr> parameters() override {
        auto p = fc1.parameters(); auto p2 = fc2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        return p;
    }
    void train() override { drop.training = true; }
    void eval() override { drop.training = false; }
};

struct TransformerBlock : Module {
    LayerNorm ln1, ln2;
    MultiHeadSelfAttention attn;
    FeedForward ff;

    TransformerBlock(int d_model, int n_heads, int d_ff, double dropout = 0.1)
        : ln1(d_model), ln2(d_model), attn(d_model, n_heads, dropout), ff(d_model, d_ff, dropout) {}

    Vec forward(const Vec& X, const std::vector<int>& pad_mask) {
        Vec ln1_out;
        for (auto& x : X) ln1_out.push_back(ln1.forward({x})[0]);
        Vec attn_out = attn.forward(ln1_out, pad_mask);
        Vec X2;
        for (size_t i = 0; i < X.size(); ++i) {
            Vec sum(X[i].size());
            for (size_t j = 0; j < X[i].size(); ++j) sum[j] = *X[i][j] + attn_out[i][j];
            X2.push_back(sum);
        }
        Vec ln2_out;
        for (auto& x : X2) ln2_out.push_back(ln2.forward({x})[0]);
        Vec ff_out = ff.forward(ln2_out);
        Vec out;
        for (size_t i = 0; i < X2.size(); ++i) {
            Vec sum(X2[i].size());
            for (size_t j = 0; j < X2[i].size(); ++j) sum[j] = *X2[i][j] + ff_out[i][j];
            out.push_back(sum);
        }
        return out;
    }

    Vec forward_incremental(const Vec& x, std::vector<Vec>& K_cache, std::vector<Vec>& V_cache) {
        Vec ln1_out = ln1.forward({x})[0];
        Vec attn_out = attn.forward_incremental(ln1_out, K_cache, V_cache);
        Vec x2;
        for (size_t i = 0; i < x.size(); ++i) x2.push_back(*x[i] + attn_out[i]);
        Vec ln2_out = ln2.forward({x2})[0];
        Vec ff_out = ff.forward({ln2_out}); // ff.forward expects list, so wrap
        Vec out;
        for (size_t i = 0; i < x2.size(); ++i) out.push_back(*x2[i] + ff_out[i]);
        return out;
    }

    std::vector<ValuePtr> parameters() override {
        auto p = ln1.parameters(); auto p2 = attn.parameters();
        auto p3 = ln2.parameters(); auto p4 = ff.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        p.insert(p.end(), p3.begin(), p3.end());
        p.insert(p.end(), p4.begin(), p4.end());
        return p;
    }
    void train() override { attn.train(); ff.train(); }
    void eval() override { attn.eval(); ff.eval(); }
};

struct MiniGPT : Module {
    int d_model, max_len, vocab_size;
    Embedding embed;
    PositionalEmbedding pos_embed;
    Dropout embed_drop;
    std::vector<TransformerBlock> blocks;
    LayerNorm ln_f;
    Linear head;
    std::vector<std::vector<Vec>> caches; // one per block: pair of K,V caches
    bool use_kv_cache;

    MiniGPT(int vocab_size, int d_model = 16, int n_heads = 2, int n_layers = 2, int d_ff = 32,
            int max_len = 64, double dropout = 0.1)
        : d_model(d_model), max_len(max_len), vocab_size(vocab_size),
          embed(vocab_size, d_model), pos_embed(max_len, d_model), embed_drop(dropout),
          ln_f(d_model), head(d_model, vocab_size), use_kv_cache(false) {
        for (int i = 0; i < n_layers; ++i)
            blocks.emplace_back(d_model, n_heads, d_ff, dropout);
        // weight tying: head.W = embed.table (we must share parameters)
        // In our design, head.W and embed.table are separate matrices.
        // We can make head.W refer to same Value objects as embed.table.
        // Better: construct head with bias only, and set W manually.
        // For simplicity, we just copy pointers? Not directly. We'll skip weight tying in C++ demo.
        // (Weight tying can be done by using embed.table as head.W, but here we keep separate.)
    }

    Vec forward(const std::vector<int>& token_ids, const std::vector<int>& pad_mask = {}) {
        Vec X;
        for (size_t pos = 0; pos < token_ids.size(); ++pos) {
            Vec emb = embed.forward(token_ids[pos]);
            Vec pe = pos_embed.forward(pos);
            Vec combined;
            for (int d = 0; d < d_model; ++d) combined.push_back(*emb[d] + pe[d]);
            X.push_back(combined);
        }
        X = embed_drop.forward(X);
        for (auto& block : blocks) X = block.forward(X, pad_mask);
        Vec ln_out;
        for (auto& x : X) ln_out.push_back(ln_f.forward({x})[0]);
        Vec logits_seq;
        for (auto& x : ln_out) logits_seq.push_back(head.forward(x)[0]); // head returns vector of logits
        return logits_seq;
    }

    void init_cache() {
        caches.clear();
        for (size_t i = 0; i < blocks.size(); ++i) {
            caches.push_back(std::vector<Vec>()); // will store as pair of vectors: K and V
        }
        use_kv_cache = true;
    }

    Vec forward_incremental(int token_id, int pos) {
        if (pos >= max_len) throw std::runtime_error("pos exceeds max_len");
        Vec emb = embed.forward(token_id);
        Vec pe = pos_embed.forward(pos);
        Vec x;
        for (int d = 0; d < d_model; ++d) x.push_back(*emb[d] + pe[d]);
        x = embed_drop.forward(x); // drops on single vector
        for (size_t i = 0; i < blocks.size(); ++i) {
            // We need separate K/V caches per block. For simplicity, use two vectors per block.
            // We'll store as [K_cache, V_cache] in caches[i].
            if (caches[i].size() < 2) {
                caches[i] = {Vec(), Vec()}; // initialize empty K and V
            }
            x = blocks[i].forward_incremental(x, caches[i][0], caches[i][1]);
        }
        Vec ln_out = ln_f.forward({x})[0];
        return head.forward(ln_out);
    }

    std::vector<ValuePtr> parameters() override {
        auto p = embed.parameters();
        auto p2 = pos_embed.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        for (auto& b : blocks) {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto p_ln = ln_f.parameters();
        p.insert(p.end(), p_ln.begin(), p_ln.end());
        // head bias only (weight tying omitted)
        for (auto& bi : head.b) p.push_back(bi);
        return p;
    }
    void train() override { embed_drop.training = true; for (auto& b : blocks) b.train(); }
    void eval() override { embed_drop.training = false; for (auto& b : blocks) b.eval(); }
};

// ============================================================
// 7. LOSS & OPTIMIZER
// ============================================================
ValuePtr cross_entropy_loss(const Vec& logits_seq, const std::vector<int>& target_ids, const std::vector<int>& pad_mask) {
    std::vector<ValuePtr> losses;
    for (size_t i = 0; i < logits_seq.size(); ++i) {
        if (pad_mask[i] == 0) continue;
        auto probs = softmax(logits_seq[i]); // returns list of ValuePtr
        losses.push_back(*probs[target_ids[i]]->log() * -1.0);
    }
    if (losses.empty()) return Value::create(0.0);
    ValuePtr total = Value::create(0.0);
    for (auto& l : losses) total = *total + *l;
    return *total / (double)losses.size();
}

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

void clip_grad_norm(std::vector<ValuePtr>& params, double max_norm) {
    double total_sq = 0.0;
    for (auto& p : params) total_sq += p->grad * p->grad;
    double norm = std::sqrt(total_sq);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (auto& p : params) p->grad *= scale;
    }
}

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
// 8. DATA HELPERS
// ============================================================
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
build_example(const std::vector<int>& ids, int seq_len, int pad_id) {
    std::vector<int> inp, tgt, mask;
    for (size_t start = 0; start + 1 < ids.size(); start += seq_len) {
        size_t chunk_len = std::min(ids.size() - start, (size_t)seq_len + 1);
        if (chunk_len < 2) continue;
        for (size_t i = 0; i < chunk_len - 1; ++i) inp.push_back(ids[start + i]);
        for (size_t i = 1; i < chunk_len; ++i) tgt.push_back(ids[start + i]);
        // pad
        int pad_needed = seq_len - inp.size();
        for (int p = 0; p < pad_needed; ++p) { inp.push_back(pad_id); tgt.push_back(pad_id); }
        mask.insert(mask.end(), inp.size() - pad_needed, 1);
        mask.insert(mask.end(), pad_needed, 0);
        return {inp, tgt, mask};
    }
    return {};
}

// ============================================================
// 9. GENERATE
// ============================================================
int sample_from_logits(const Vec& logits, double temperature, int top_k, double top_p) {
    std::vector<double> scaled;
    for (auto& v : logits) scaled.push_back(v->data / std::max(temperature, 1e-6));
    double m = *std::max_element(scaled.begin(), scaled.end());
    std::vector<double> exps;
    for (auto& s : scaled) exps.push_back(std::exp(s - m));
    double sum_e = 0.0;
    for (auto e : exps) sum_e += e;
    std::vector<double> probs(exps.size());
    for (size_t i = 0; i < exps.size(); ++i) probs[i] = exps[i] / sum_e;

    if (top_k > 0) {
        std::vector<size_t> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        for (size_t i = top_k; i < idx.size(); ++i) probs[idx[i]] = 0.0;
    }
    if (top_p > 0) {
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
    double sum_p = 0.0;
    for (auto p : probs) sum_p += p;
    if (sum_p <= 0) return std::max_element(scaled.begin(), scaled.end()) - scaled.begin();
    for (auto& p : probs) p /= sum_p;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0, 1);
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
    model.eval();
    model.init_cache();
    auto ids = tok.encode(prompt, true, false);
    int eos_id = tok.vocab["<eos>"];
    Vec logits;
    for (size_t pos = 0; pos < ids.size(); ++pos) {
        logits = model.forward_incremental(ids[pos], pos);
    }
    for (int _ = 0; _ < max_new_tokens; ++_) {
        int next_id = sample_from_logits(logits, temperature, top_k, top_p);
        ids.push_back(next_id);
        if (next_id == eos_id) break;
        int pos = ids.size() - 1;
        if (pos >= model.max_len) break;
        logits = model.forward_incremental(next_id, pos);
    }
    model.train();
    return tok.decode(ids);
}

// ============================================================
// 10. PYTHON BINDINGS
// ============================================================
namespace py = pybind11;

PYBIND11_MODULE(minigpt, m) {
    m.doc() = "Mini-GPT C++ implementation with autograd and BPE tokenizer";

    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init<double>())
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def("backward", &Value::backward)
        .def("__repr__", &Value::repr)
        .def("__add__", [](Value& self, std::shared_ptr<Value> other) { return self + other; })
        .def("__add__", [](Value& self, double other) { return self + other; })
        .def("__mul__", [](Value& self, std::shared_ptr<Value> other) { return self * other; })
        .def("__mul__", [](Value& self, double other) { return self * other; })
        .def("__sub__", [](Value& self, std::shared_ptr<Value> other) { return self - other; })
        .def("__truediv__", [](Value& self, std::shared_ptr<Value> other) { return self / other; })
        .def("__pow__", &Value::pow)
        .def("exp", &Value::exp)
        .def("log", &Value::log)
        .def("sqrt", &Value::sqrt)
        .def("tanh", &Value::tanh)
        .def("relu", &Value::relu)
        .def("gelu", &Value::gelu);

    py::class_<ByteLevelBPETokenizer>(m, "ByteLevelBPETokenizer")
        .def(py::init<>())
        .def("train", &ByteLevelBPETokenizer::train)
        .def("encode", &ByteLevelBPETokenizer::encode,
             py::arg("text"), py::arg("add_bos")=false, py::arg("add_eos")=false)
        .def("decode", &ByteLevelBPETokenizer::decode)
        .def("save", &ByteLevelBPETokenizer::save)
        .def("load", &ByteLevelBPETokenizer::load)
        .def_readwrite("vocab", &ByteLevelBPETokenizer::vocab)
        .def_readwrite("merge_order", &ByteLevelBPETokenizer::merge_order);

    py::class_<Dropout>(m, "Dropout")
        .def(py::init<double>())
        .def("forward", &Dropout::forward)
        .def_readwrite("training", &Dropout::training);

    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, bool>())
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);

    py::class_<Embedding>(m, "Embedding")
        .def(py::init<int, int, double>())
        .def("forward", &Embedding::forward)
        .def("parameters", &Embedding::parameters);

    py::class_<PositionalEmbedding>(m, "PositionalEmbedding")
        .def(py::init<int, int, double>())
        .def("forward", &PositionalEmbedding::forward)
        .def("parameters", &PositionalEmbedding::parameters);

    py::class_<LayerNorm>(m, "LayerNorm")
        .def(py::init<int, double>())
        .def("forward", &LayerNorm::forward)
        .def("parameters", &LayerNorm::parameters);

    py::class_<MultiHeadSelfAttention>(m, "MultiHeadSelfAttention")
        .def(py::init<int, int, double>())
        .def("forward", &MultiHeadSelfAttention::forward)
        .def("forward_incremental", &MultiHeadSelfAttention::forward_incremental)
        .def("parameters", &MultiHeadSelfAttention::parameters)
        .def("train", &MultiHeadSelfAttention::train)
        .def("eval", &MultiHeadSelfAttention::eval);

    py::class_<FeedForward>(m, "FeedForward")
        .def(py::init<int, int, double>())
        .def("forward", &FeedForward::forward)
        .def("parameters", &FeedForward::parameters)
        .def("train", &FeedForward::train)
        .def("eval", &FeedForward::eval);

    py::class_<TransformerBlock>(m, "TransformerBlock")
        .def(py::init<int, int, int, double>())
        .def("forward", &TransformerBlock::forward)
        .def("forward_incremental", &TransformerBlock::forward_incremental)
        .def("parameters", &TransformerBlock::parameters)
        .def("train", &TransformerBlock::train)
        .def("eval", &TransformerBlock::eval);

    py::class_<MiniGPT>(m, "MiniGPT")
        .def(py::init<int, int, int, int, int, int, double>())
        .def("forward", &MiniGPT::forward)
        .def("init_cache", &MiniGPT::init_cache)
        .def("forward_incremental", &MiniGPT::forward_incremental)
        .def("parameters", &MiniGPT::parameters)
        .def("train", &MiniGPT::train)
        .def("eval", &MiniGPT::eval);

    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<ValuePtr>, double, double, double, double, double>())
        .def("step", &AdamW::step)
        .def("zero_grad", &AdamW::zero_grad);

    py::class_<WarmupCosineScheduler>(m, "WarmupCosineScheduler")
        .def(py::init<AdamW*, int, int, double, double>())
        .def("step", &WarmupCosineScheduler::step);

    m.def("clip_grad_norm", &clip_grad_norm);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("build_example", &build_example);
    m.def("generate", &generate);

    // Init byte decoder before any tokenizer usage
    init_byte_decoder();
}