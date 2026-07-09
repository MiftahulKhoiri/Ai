#include "tokenizer.h"
#include <regex>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Static byte maps
static std::unordered_map<int, std::string> BYTE_ENCODER;
static std::unordered_map<std::string, int> BYTE_DECODER;
static bool byte_maps_initialized = false;

void ByteLevelBPETokenizer::init_byte_maps() {
    if (byte_maps_initialized) return;
    byte_maps_initialized = true;
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

void ByteLevelBPETokenizer::train(const std::string& corpus, int vocab_size) {
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

std::vector<std::string> ByteLevelBPETokenizer::apply_bpe(const std::vector<std::string>& symbols) {
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

std::vector<int> ByteLevelBPETokenizer::encode(const std::string& text, bool add_bos, bool add_eos) {
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

std::string ByteLevelBPETokenizer::decode(const std::vector<int>& ids) {
    std::string bytes;
    for (int id : ids) {
        auto it = inv_vocab.find(id);
        if (it != inv_vocab.end()) {
            std::string tok = it->second;
            if (tok == "<pad>" || tok == "<bos>" || tok == "<eos>" || tok == "<unk>")
                continue;
            for (char c : tok) {
                std::string key(1, c);
                if (BYTE_DECODER.count(key))
                    bytes.push_back(static_cast<char>(BYTE_DECODER[key]));
            }
        }
    }
    return bytes;
}

void ByteLevelBPETokenizer::save(const std::string& path) {
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

void ByteLevelBPETokenizer::load(const std::string& path) {
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