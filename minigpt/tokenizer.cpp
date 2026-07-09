#include "tokenizer.h"
#include <regex>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

namespace py = pybind11;

// Static byte maps
static std::unordered_map<int, std::string> BYTE_ENCODER;
static std::unordered_map<std::string, int> BYTE_DECODER;
static bool byte_maps_initialized = false;

void ByteLevelBPETokenizer::init_byte_maps() {
    if (byte_maps_initialized) return;
    byte_maps_initialized = true;
    
    // Karakter yang bisa dicetak langsung (valid ASCII/Unicode)
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);           // 33-126
    for (int b = 0xa1; b <= 0xac; ++b) bs.push_back(b);         // 161-172
    for (int b = 0xae; b <= 0xff; ++b) bs.push_back(b);         // 174-255
    
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            // Gunakan Unicode code point U+0100 sampai U+01FF (Latin Extended-A)
            // Ini adalah karakter valid UTF-8 yang aman digunakan
            int code_point = 0x0100 + n;
            
            // Encode code point ke UTF-8
            std::string utf8_char;
            if (code_point < 0x80) {
                utf8_char += static_cast<char>(code_point);
            } else if (code_point < 0x800) {
                utf8_char += static_cast<char>(0xC0 | (code_point >> 6));
                utf8_char += static_cast<char>(0x80 | (code_point & 0x3F));
            } else if (code_point < 0x10000) {
                utf8_char += static_cast<char>(0xE0 | (code_point >> 12));
                utf8_char += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (code_point & 0x3F));
            }
            
            BYTE_ENCODER[b] = utf8_char;
            ++n;
        } else {
            // Karakter ASCII yang bisa langsung digunakan
            BYTE_ENCODER[b] = std::string(1, static_cast<char>(b));
        }
    }
    
    // Build decoder (reverse mapping)
    BYTE_DECODER.clear();
    for (auto& p : BYTE_ENCODER) {
        BYTE_DECODER[p.second] = p.first;
    }
}

static const std::regex PRETOKEN_PAT(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)"
);

void ByteLevelBPETokenizer::train(const std::string& corpus, int vocab_size) {
    init_byte_maps();
    
    // Tokenisasi teks menjadi simbol byte-level
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

    // Hitung frekuensi setiap kata (urutan simbol)
    std::map<std::vector<std::string>, int> freq;
    for (auto& w : words) freq[w]++;

    // Inisialisasi splits: setiap kata dipecah menjadi simbol-simbol
    std::map<std::vector<std::string>, std::vector<std::string>> splits;
    for (auto& p : freq) splits[p.first] = p.first;

    // Base vocabulary dari 256 byte
    std::set<std::string> base_vocab;
    for (auto& p : BYTE_ENCODER) base_vocab.insert(p.second);

    int num_merges = std::max(0, vocab_size - (int)base_vocab.size() - 4);  // 4 special tokens
    
    for (int step = 0; step < num_merges; ++step) {
        // Hitung frekuensi setiap pasangan
        std::map<std::pair<std::string, std::string>, int> pair_counts;
        for (auto& wf : freq) {
            auto& syms = splits[wf.first];
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                pair_counts[{syms[i], syms[i+1]}] += wf.second;
            }
        }
        
        if (pair_counts.empty()) break;
        
        // Cari pasangan paling sering
        auto best = std::max_element(pair_counts.begin(), pair_counts.end(),
            [](auto& a, auto& b) { return a.second < b.second; });
        
        if (best->second < 2) break;  // Berhenti jika frekuensi terlalu rendah
        
        std::pair<std::string, std::string> best_pair = best->first;
        std::string merged = best_pair.first + best_pair.second;
        merge_order.push_back(best_pair);

        // Merge pasangan di semua kata
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

    // Build vocabulary
    std::vector<std::string> specials = {"<pad>", "<bos>", "<eos>", "<unk>"};
    std::vector<std::string> all_tokens = specials;
    
    // Tambahkan base vocab
    for (auto& s : base_vocab) all_tokens.push_back(s);
    
    // Tambahkan merged tokens
    for (auto& p : merge_order) {
        std::string merged = p.first + p.second;
        if (std::find(all_tokens.begin(), all_tokens.end(), merged) == all_tokens.end()) {
            all_tokens.push_back(merged);
        }
    }
    
    // Sort dan hapus duplikat (kecuali special tokens di awal)
    std::sort(all_tokens.begin() + 4, all_tokens.end());
    all_tokens.erase(std::unique(all_tokens.begin() + 4, all_tokens.end()), all_tokens.end());
    
    // Buat vocabulary mapping
    vocab.clear();
    inv_vocab.clear();
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        vocab[all_tokens[i]] = i;
        inv_vocab[i] = all_tokens[i];
    }
    
    // Buat merge rank
    merge_rank.clear();
    for (size_t i = 0; i < merge_order.size(); ++i) {
        merge_rank[merge_order[i]] = i;
    }
}

std::vector<std::string> ByteLevelBPETokenizer::apply_bpe(const std::vector<std::string>& symbols) {
    std::vector<std::string> word = symbols;
    
    if (word.size() <= 1) return word;
    
    while (true) {
        // Cari pasangan dengan rank terendah (prioritas tertinggi)
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
        
        if (word.size() == 1) break;
    }
    
    return word;
}

std::vector<int> ByteLevelBPETokenizer::encode(const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;
    
    if (add_bos) {
        auto it = vocab.find("<bos>");
        if (it != vocab.end()) ids.push_back(it->second);
    }
    
    std::sregex_iterator it(text.begin(), text.end(), PRETOKEN_PAT);
    std::sregex_iterator end;
    
    for (; it != end; ++it) {
        std::string token = it->str();
        std::vector<std::string> syms;
        
        // Konversi setiap byte ke representasi Unicode
        for (unsigned char c : token) {
            auto enc_it = BYTE_ENCODER.find(c);
            if (enc_it != BYTE_ENCODER.end()) {
                syms.push_back(enc_it->second);
            }
        }
        
        // Apply BPE
        std::vector<std::string> bpe_result = apply_bpe(syms);
        
        for (auto& s : bpe_result) {
            auto vit = vocab.find(s);
            if (vit != vocab.end()) {
                ids.push_back(vit->second);
            } else {
                // Fallback ke <unk>
                auto unk_it = vocab.find("<unk>");
                if (unk_it != vocab.end()) {
                    ids.push_back(unk_it->second);
                }
            }
        }
    }
    
    if (add_eos) {
        auto it = vocab.find("<eos>");
        if (it != vocab.end()) ids.push_back(it->second);
    }
    
    return ids;
}

std::string ByteLevelBPETokenizer::decode(const std::vector<int>& ids) {
    std::vector<uint8_t> byte_sequence;
    
    for (int id : ids) {
        auto it = inv_vocab.find(id);
        if (it == inv_vocab.end()) continue;
        
        std::string tok = it->second;
        
        // Skip special tokens
        if (tok == "<pad>" || tok == "<bos>" || tok == "<eos>" || tok == "<unk>")
            continue;
        
        // Konversi setiap karakter Unicode kembali ke byte asli
        for (size_t i = 0; i < tok.size(); ) {
            unsigned char c = tok[i];
            int code_point = 0;
            int bytes_count = 0;
            
            // Decode UTF-8 character
            if ((c & 0x80) == 0) {
                // Single byte (ASCII)
                code_point = c;
                bytes_count = 1;
            } else if ((c & 0xE0) == 0xC0) {
                // Two bytes
                code_point = c & 0x1F;
                bytes_count = 2;
            } else if ((c & 0xF0) == 0xE0) {
                // Three bytes
                code_point = c & 0x0F;
                bytes_count = 3;
            } else if ((c & 0xF8) == 0xF0) {
                // Four bytes
                code_point = c & 0x07;
                bytes_count = 4;
            } else {
                // Invalid UTF-8, skip
                i++;
                continue;
            }
            
            // Extract continuation bytes
            for (int j = 1; j < bytes_count; j++) {
                if (i + j >= tok.size()) break;
                code_point = (code_point << 6) | (tok[i + j] & 0x3F);
            }
            
            // Kembalikan ke byte asli
            if (code_point >= 0x0100 && code_point < 0x0200) {
                // Karakter yang kita encode dari byte (U+0100 - U+01FF)
                byte_sequence.push_back(static_cast<uint8_t>(code_point - 0x0100));
            } else if (code_point < 0x80) {
                // ASCII character langsung
                byte_sequence.push_back(static_cast<uint8_t>(code_point));
            } else if (code_point >= 0xA0 && code_point <= 0xFF) {
                // Latin-1 Supplement yang valid
                byte_sequence.push_back(static_cast<uint8_t>(code_point));
            }
            
            i += bytes_count;
        }
    }
    
    // Konversi byte sequence ke string UTF-8
    return std::string(byte_sequence.begin(), byte_sequence.end());
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
    if (f.is_open()) {
        f << dumped;
        f.close();
    }
}

void ByteLevelBPETokenizer::load(const std::string& path) {
    py::module_ json = py::module_::import("json");
    
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::stringstream buffer;
    buffer << f.rdbuf();
    f.close();
    
    py::dict data = json.attr("loads")(buffer.str()).cast<py::dict>();
    
    // Load vocab
    vocab.clear();
    inv_vocab.clear();
    
    // Handle vocab loading dengan benar
    py::dict vocab_dict = data["vocab"].cast<py::dict>();
    for (auto item : vocab_dict) {
        std::string token = item.first.cast<std::string>();
        int id = item.second.cast<int>();
        vocab[token] = id;
        inv_vocab[id] = token;
    }
    
    // Load merge order
    merge_order.clear();
    merge_rank.clear();
    
    std::vector<py::list> mo = data["merge_order"].cast<std::vector<py::list>>();
    for (auto& t : mo) {
        std::string a = t[0].cast<std::string>();
        std::string b = t[1].cast<std::string>();
        merge_order.push_back({a, b});
    }
    
    for (size_t i = 0; i < merge_order.size(); ++i) {
        merge_rank[merge_order[i]] = i;
    }
}