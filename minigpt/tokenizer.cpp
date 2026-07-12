// tokenizer.cpp
#include "tokenizer.h"
#include <regex>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <array>
#include <unordered_set>
#include <iostream>
#include <cmath>

// Inisialisasi static members
bool ByteLevelBPETokenizer::byte_maps_initialized = false;
std::array<std::string, 256> ByteLevelBPETokenizer::BYTE_ENCODER;
std::unordered_map<std::string, int> ByteLevelBPETokenizer::BYTE_DECODER;

// Implementasi custom hash
std::size_t pair_hash::operator()(const std::pair<std::string, std::string>& p) const noexcept {
    auto h1 = std::hash<std::string>{}(p.first);
    auto h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 << 1);
}

std::size_t vector_hash::operator()(const std::vector<std::string>& v) const noexcept {
    std::size_t seed = v.size();
    for (const auto& s : v) {
        seed ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

static const std::regex PRETOKEN_PAT(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)"
);

ByteLevelBPETokenizer::ByteLevelBPETokenizer() {
    init_byte_maps();
}

std::string ByteLevelBPETokenizer::utf8_encode(uint32_t codepoint) noexcept {
    std::string result;
    if (codepoint <= 0x7F) {
        result += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7FF) {
        result += static_cast<char>(0xC0 | (codepoint >> 6));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (codepoint >> 12));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint <= 0x10FFFF) {
        result += static_cast<char>(0xF0 | (codepoint >> 18));
        result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    return result;
}

void ByteLevelBPETokenizer::init_byte_maps() {
    if (byte_maps_initialized) return;
    byte_maps_initialized = true;

    std::unordered_set<int> printable_bytes;
    printable_bytes.reserve(256);

    for (int b = '!'; b <= '~'; ++b) printable_bytes.insert(b);
    for (int b = 0xA1; b <= 0xAC; ++b) printable_bytes.insert(b);
    for (int b = 0xAE; b <= 0xFF; ++b) printable_bytes.insert(b);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (printable_bytes.count(b)) {
            uint32_t codepoint = b;
            BYTE_ENCODER[b] = utf8_encode(codepoint);
        } else {
            uint32_t codepoint = 0xE000 + n;
            BYTE_ENCODER[b] = utf8_encode(codepoint);
            n++;
        }
    }

    BYTE_DECODER.clear();
    BYTE_DECODER.reserve(256);
    for (int b = 0; b < 256; ++b) {
        BYTE_DECODER.emplace(BYTE_ENCODER[b], b);
    }
}

bool ByteLevelBPETokenizer::is_special_token(const std::string& token) const noexcept {
    return token == PAD_TOKEN || token == BOS_TOKEN || 
           token == EOS_TOKEN || token == UNK_TOKEN;
}

std::string ByteLevelBPETokenizer::escape_json(const std::string& str) const {
    std::string escaped;
    escaped.reserve(str.size() * 2);

    for (char c : str) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n";  break;
            case '\t': escaped += "\\t";  break;
            case '\r': escaped += "\\r";  break;
            case '\b': escaped += "\\b";  break;
            case '\f': escaped += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    escaped += buf;
                } else {
                    escaped += c;
                }
        }
    }
    return escaped;
}

// ============================================================
// TRAIN
// ============================================================
void ByteLevelBPETokenizer::train(const std::string& corpus, int vocab_size) {
    if (corpus.empty()) {
        std::cerr << "⚠️ Warning: Corpus kosong, menggunakan vocab minimal" << std::endl;
        vocab.clear();
        inv_vocab.clear();
        const std::array<std::string, SPECIAL_TOKEN_COUNT> specials = {
            PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
        };
        for (size_t i = 0; i < specials.size(); ++i) {
            vocab[specials[i]] = static_cast<int>(i);
            inv_vocab[static_cast<int>(i)] = specials[i];
        }
        return;
    }

    if (vocab_size <= SPECIAL_TOKEN_COUNT) {
        std::cerr << "⚠️ Warning: vocab_size terlalu kecil, menggunakan minimal" << std::endl;
        vocab_size = SPECIAL_TOKEN_COUNT + 10;
    }

    std::vector<std::vector<std::string>> words;
    std::sregex_iterator it(corpus.begin(), corpus.end(), PRETOKEN_PAT);
    std::sregex_iterator end;

    for (; it != end; ++it) {
        const std::string& token = it->str();
        if (token.empty()) continue;

        std::vector<std::string> syms;
        syms.reserve(token.size());

        for (unsigned char c : token) {
            syms.emplace_back(BYTE_ENCODER[c]);
        }

        if (!syms.empty()) {
            words.emplace_back(std::move(syms));
        }
    }

    if (words.empty()) {
        std::cerr << "⚠️ Warning: Tidak ada token yang valid, menggunakan vocab minimal" << std::endl;
        vocab.clear();
        inv_vocab.clear();
        const std::array<std::string, SPECIAL_TOKEN_COUNT> specials = {
            PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
        };
        for (size_t i = 0; i < specials.size(); ++i) {
            vocab[specials[i]] = static_cast<int>(i);
            inv_vocab[static_cast<int>(i)] = specials[i];
        }
        return;
    }

    std::unordered_map<std::vector<std::string>, int, vector_hash> freq;
    for (auto& w : words) {
        freq[std::move(w)]++;
    }

    std::unordered_map<std::vector<std::string>, std::vector<std::string>, vector_hash> splits;
    for (const auto& p : freq) {
        splits.emplace(p.first, p.first);
    }

    std::set<std::string> base_vocab;
    for (const auto& enc : BYTE_ENCODER) {
        base_vocab.insert(enc);
    }

    int num_merges = std::max(0, vocab_size - static_cast<int>(base_vocab.size()) - SPECIAL_TOKEN_COUNT);
    merge_order.clear();
    merge_order.reserve(num_merges);
    merge_rank.clear();
    merge_rank.reserve(num_merges);

    for (int step = 0; step < num_merges; ++step) {
        std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> pair_counts;
        pair_counts.reserve(freq.size() * 5);

        for (const auto& wf : freq) {
            const auto& syms = splits[wf.first];
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                pair_counts[{syms[i], syms[i+1]}] += wf.second;
            }
        }

        if (pair_counts.empty()) break;

        // FIX: tambah tie-breaker deterministik. Sebelumnya kalau ada
        // beberapa pair dengan frekuensi sama persis, max_element memilih
        // yang mana saja tergantung urutan iterasi unordered_map (beda
        // tiap run walau corpus sama) -> hasil training tokenizer tidak
        // reproducible. Sekarang kalau frekuensi sama, menangkan pair yang
        // secara leksikografis lebih besar -- aturan tetap, hasil selalu
        // sama untuk corpus yang sama.
        auto best = std::max_element(pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second < b.second;
                return a.first > b.first;
            });

        if (best->second < 2) break;

        const auto& best_pair = best->first;
        std::string merged = best_pair.first + best_pair.second;
        merge_order.emplace_back(best_pair);
        merge_rank[best_pair] = step;

        for (auto& wf : freq) {
            auto& syms = splits[wf.first];
            std::vector<std::string> new_syms;
            new_syms.reserve(syms.size());

            for (size_t i = 0; i < syms.size(); ) {
                if (i + 1 < syms.size() && 
                    syms[i] == best_pair.first && 
                    syms[i+1] == best_pair.second) {
                    new_syms.emplace_back(merged);
                    i += 2;
                } else {
                    new_syms.emplace_back(std::move(syms[i]));
                    i++;
                }
            }
            syms = std::move(new_syms);
        }
    }

    std::vector<std::string> specials = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN};
    std::vector<std::string> all_tokens;
    all_tokens.reserve(specials.size() + base_vocab.size() + merge_order.size());
    all_tokens = specials;

    for (const auto& s : base_vocab) {
        all_tokens.emplace_back(s);
    }

    std::unordered_set<std::string> merged_set;
    merged_set.reserve(merge_order.size());
    for (const auto& p : merge_order) {
        merged_set.emplace(p.first + p.second);
    }
    for (const auto& s : merged_set) {
        all_tokens.emplace_back(s);
    }

    std::sort(all_tokens.begin() + SPECIAL_TOKEN_COUNT, all_tokens.end());
    all_tokens.erase(
        std::unique(all_tokens.begin() + SPECIAL_TOKEN_COUNT, all_tokens.end()),
        all_tokens.end()
    );

    vocab.clear();
    inv_vocab.clear();
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        vocab[all_tokens[i]] = static_cast<int>(i);
        inv_vocab[static_cast<int>(i)] = all_tokens[i];
    }
}

// ============================================================
// APPLY BPE
// ============================================================
std::vector<std::string> ByteLevelBPETokenizer::apply_bpe(const std::vector<std::string>& symbols) {
    if (symbols.size() <= 1) return symbols;

    std::vector<std::string> word = symbols;

    while (true) {
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
        new_word.reserve(word.size());

        for (size_t i = 0; i < word.size(); ) {
            if (i + 1 < word.size() && 
                word[i] == best_pair.first && 
                word[i+1] == best_pair.second) {
                new_word.emplace_back(merged);
                i += 2;
            } else {
                new_word.emplace_back(std::move(word[i]));
                i++;
            }
        }

        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    return word;
}

// ============================================================
// ENCODE
// ============================================================
std::vector<int> ByteLevelBPETokenizer::encode(const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;
    ids.reserve(text.size() / 2 + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));

    if (add_bos) {
        auto it = vocab.find(BOS_TOKEN);
        if (it != vocab.end()) ids.emplace_back(it->second);
    }

    std::sregex_iterator it(text.begin(), text.end(), PRETOKEN_PAT);
    std::sregex_iterator end;

    for (; it != end; ++it) {
        const std::string& token = it->str();
        if (token.empty()) continue;

        std::vector<std::string> syms;
        syms.reserve(token.size());

        for (unsigned char c : token) {
            syms.emplace_back(BYTE_ENCODER[c]);
        }

        auto bpe_result = apply_bpe(syms);
        for (const auto& s : bpe_result) {
            auto vit = vocab.find(s);
            if (vit != vocab.end()) {
                ids.emplace_back(vit->second);
            } else {
                auto unk_it = vocab.find(UNK_TOKEN);
                if (unk_it != vocab.end()) {
                    ids.emplace_back(unk_it->second);
                }
            }
        }
    }

    if (add_eos) {
        auto eos_it = vocab.find(EOS_TOKEN);
        if (eos_it != vocab.end()) ids.emplace_back(eos_it->second);
    }

    return ids;
}

// ============================================================
// DECODE
// ============================================================
std::string ByteLevelBPETokenizer::decode(const std::vector<int>& ids) {
    std::vector<uint8_t> byte_sequence;
    byte_sequence.reserve(ids.size() * 2);

    for (int id : ids) {
        auto it = inv_vocab.find(id);
        if (it == inv_vocab.end()) continue;

        const std::string& tok = it->second;
        if (is_special_token(tok)) continue;

        for (size_t i = 0; i < tok.size(); ) {
            int bytes_consumed = 0;
            uint32_t codepoint = decode_utf8_char(tok, i, bytes_consumed);

            if (bytes_consumed > 0) {
                if (codepoint >= 0xE000 && codepoint <= 0xE0FF) {
                    byte_sequence.emplace_back(static_cast<uint8_t>(codepoint - 0xE000));
                } else if (codepoint < 0x80) {
                    byte_sequence.emplace_back(static_cast<uint8_t>(codepoint));
                } else if (codepoint >= 0xA0 && codepoint <= 0xFF) {
                    byte_sequence.emplace_back(static_cast<uint8_t>(codepoint));
                } else if (codepoint == 0xFFFD) {
                    // Skip invalid character
                } else {
                    std::string char_str = tok.substr(i, bytes_consumed);
                    auto dec_it = BYTE_DECODER.find(char_str);
                    if (dec_it != BYTE_DECODER.end()) {
                        byte_sequence.emplace_back(static_cast<uint8_t>(dec_it->second));
                    }
                }
                i += bytes_consumed;
            } else {
                i++;
            }
        }
    }

    return std::string(byte_sequence.begin(), byte_sequence.end());
}

// ============================================================
// SAVE
// ============================================================
bool ByteLevelBPETokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "❌ Error: Cannot open file for writing: " << path << std::endl;
        return false;
    }

    f << "{\n";

    f << "  \"vocab\": {\n";
    size_t count = 0;
    for (const auto& p : vocab) {
        if (count > 0) f << ",\n";
        f << "    \"" << escape_json(p.first) << "\": " << p.second;
        count++;
    }
    f << "\n  },\n";

    f << "  \"merge_order\": [\n";
    count = 0;
    for (const auto& p : merge_order) {
        if (count > 0) f << ",\n";
        f << "    [\"" << escape_json(p.first) << "\", \"" << escape_json(p.second) << "\"]";
        count++;
    }
    f << "\n  ]\n";

    f << "}\n";
    f.close();

    if (!f.good()) {
        std::cerr << "❌ Error: Failed to write tokenizer file: " << path << std::endl;
        return false;
    }

    std::cout << "✅ Tokenizer saved to: " << path << std::endl;
    return true;
}

// ============================================================
// LOAD
// ============================================================
bool ByteLevelBPETokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "❌ Error: Cannot open file for reading: " << path << std::endl;
        return false;
    }

    std::string content;
    f.seekg(0, std::ios::end);
    content.reserve(f.tellg());
    f.seekg(0, std::ios::beg);
    content.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    f.close();

    if (content.empty()) {
        std::cerr << "❌ Error: Empty tokenizer file: " << path << std::endl;
        return false;
    }

    vocab.clear();
    inv_vocab.clear();
    merge_order.clear();
    merge_rank.clear();

    auto find_json_string = [](const std::string& str, size_t start, std::string& result) -> size_t {
        if (start >= str.size() || str[start] != '"') return std::string::npos;

        result.clear();
        size_t i = start + 1;
        while (i < str.size()) {
            if (str[i] == '\\' && i + 1 < str.size()) {
                i++;
                switch (str[i]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u':
                        if (i + 4 < str.size()) {
                            std::string hex = str.substr(i + 1, 4);
                            uint16_t code = static_cast<uint16_t>(std::stoi(hex, nullptr, 16));
                            if (code < 0x80) result += static_cast<char>(code);
                            else if (code < 0x800) {
                                result += static_cast<char>(0xC0 | (code >> 6));
                                result += static_cast<char>(0x80 | (code & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (code >> 12));
                                result += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (code & 0x3F));
                            }
                            i += 4;
                        }
                        break;
                    default: result += str[i]; break;
                }
            } else if (str[i] == '"') {
                return i + 1;
            } else {
                result += str[i];
            }
            i++;
        }
        return std::string::npos;
    };

    size_t pos = content.find("\"vocab\"");
    if (pos != std::string::npos) {
        pos = content.find('{', pos);
        if (pos != std::string::npos) {
            pos++;
            while (pos < content.size()) {
                while (pos < content.size() && std::isspace(content[pos])) pos++;
                if (pos >= content.size() || content[pos] == '}') break;

                std::string key;
                pos = find_json_string(content, pos, key);
                if (pos == std::string::npos) break;

                while (pos < content.size() && (std::isspace(content[pos]) || content[pos] == ':')) pos++;

                std::string val_str;
                while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '-')) {
                    val_str += content[pos++];
                }

                if (!key.empty() && !val_str.empty()) {
                    int id = std::stoi(val_str);
                    vocab[key] = id;
                    inv_vocab[id] = key;
                }

                while (pos < content.size() && content[pos] != ',' && content[pos] != '}') pos++;
                if (pos < content.size() && content[pos] == ',') pos++;
            }
        }
    }

    pos = content.find("\"merge_order\"");
    if (pos != std::string::npos) {
        pos = content.find('[', pos);
        if (pos != std::string::npos) {
            pos++;
            while (pos < content.size()) {
                while (pos < content.size() && std::isspace(content[pos])) pos++;
                if (pos >= content.size() || content[pos] == ']') break;

                while (pos < content.size() && content[pos] != '[' && content[pos] != ']') pos++;
                if (pos >= content.size() || content[pos] == ']') break;
                pos++;

                std::string first, second;
                pos = find_json_string(content, pos, first);
                if (pos == std::string::npos) break;

                while (pos < content.size() && content[pos] != ',' && content[pos] != ']') pos++;
                if (pos < content.size() && content[pos] == ',') pos++;

                pos = find_json_string(content, pos, second);
                if (pos == std::string::npos) break;

                if (!first.empty() && !second.empty()) {
                    merge_order.emplace_back(first, second);
                    merge_rank[{first, second}] = merge_order.size() - 1;
                }

                while (pos < content.size() && content[pos] != ']' && content[pos] != '[') pos++;
                if (pos < content.size() && content[pos] == ']') pos++;

                while (pos < content.size() && content[pos] != ',' && content[pos] != ']') pos++;
                if (pos < content.size() && content[pos] == ',') pos++;
            }
        }
    }

    if (vocab.empty()) {
        std::cerr << "❌ Error: Failed to parse tokenizer file: " << path << std::endl;
        return false;
    }

    std::cout << "✅ Tokenizer loaded from: " << path << " (vocab size: " << vocab.size() << ")" << std::endl;
    return true;
}

// ============================================================
// DECODE UTF8 CHAR
// ============================================================
uint32_t decode_utf8_char(const std::string& str, size_t& i, int& bytes_consumed) noexcept {
    bytes_consumed = 0;
    if (i >= str.size()) return 0;

    unsigned char c = str[i];
    uint32_t codepoint = 0;
    int seq_len = 0;

    if ((c & 0x80) == 0) {
        codepoint = c;
        seq_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
        if (c < 0xC2) {
            bytes_consumed = 1;
            return 0xFFFD;
        }
        codepoint = c & 0x1F;
        seq_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        codepoint = c & 0x0F;
        seq_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        if (c > 0xF4) {
            bytes_consumed = 1;
            return 0xFFFD;
        }
        codepoint = c & 0x07;
        seq_len = 4;
    } else {
        bytes_consumed = 1;
        return 0xFFFD;
    }

    for (int j = 1; j < seq_len; j++) {
        if (i + j >= str.size()) {
            bytes_consumed = j;
            return 0xFFFD;
        }
        unsigned char cont = str[i + j];
        if ((cont & 0xC0) != 0x80) {
            bytes_consumed = j;
            return 0xFFFD;
        }
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }

    if ((seq_len == 2 && codepoint < 0x80) ||
        (seq_len == 3 && codepoint < 0x800) ||
        (seq_len == 4 && codepoint < 0x10000)) {
        bytes_consumed = seq_len;
        return 0xFFFD;
    }

    if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
        bytes_consumed = seq_len;
        return 0xFFFD;
    }

    if (codepoint > 0x10FFFF) {
        bytes_consumed = seq_len;
        return 0xFFFD;
    }

    bytes_consumed = seq_len;
    return codepoint;
}