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

// Inisialisasi static members
bool ByteLevelBPETokenizer::byte_maps_initialized = false;
std::array<std::string, 256> ByteLevelBPETokenizer::BYTE_ENCODER;
std::unordered_map<std::string, int> ByteLevelBPETokenizer::BYTE_DECODER;

// Custom hash untuk pair<string, string>
struct pair_hash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Custom hash untuk vector<string>
struct vector_hash {
    std::size_t operator()(const std::vector<std::string>& v) const noexcept {
        std::size_t seed = v.size();
        for (const auto& s : v) {
            seed ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

static const std::regex PRETOKEN_PAT(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)"
);

// Special tokens sebagai constexpr
static constexpr const char* PAD_TOKEN = "<pad>";
static constexpr const char* BOS_TOKEN = "<bos>";
static constexpr const char* EOS_TOKEN = "<eos>";
static constexpr const char* UNK_TOKEN = "<unk>";
static constexpr int SPECIAL_TOKEN_COUNT = 4;

ByteLevelBPETokenizer::ByteLevelBPETokenizer() {
    init_byte_maps();
}

std::string ByteLevelBPETokenizer::utf8_encode(uint32_t codepoint) noexcept {
    std::string result;
    if (codepoint <= 0x7F) {
        // 1 byte (ASCII)
        result += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7FF) {
        // 2 bytes
        result += static_cast<char>(0xC0 | (codepoint >> 6));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint <= 0xFFFF) {
        // 3 bytes
        result += static_cast<char>(0xE0 | (codepoint >> 12));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint <= 0x10FFFF) {
        // 4 bytes
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

    // Kumpulkan byte yang bisa dicetak langsung sebagai karakter ASCII/Latin-1
    std::vector<int> printable_bytes;
    printable_bytes.reserve(256);
    
    // ASCII printable (33-126)
    for (int b = '!'; b <= '~'; ++b) printable_bytes.emplace_back(b);
    // Latin-1 Supplement printable (161-172, 174-255)
    for (int b = 0xA1; b <= 0xAC; ++b) printable_bytes.emplace_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) printable_bytes.emplace_back(b);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(printable_bytes.begin(), printable_bytes.end(), b) != printable_bytes.end()) {
            // Byte yang printable: gunakan langsung sebagai karakter Latin-1
            uint32_t codepoint;
            if (b < 0x80) {
                codepoint = b;  // ASCII
            } else {
                codepoint = b;  // Latin-1
            }
            BYTE_ENCODER[b] = utf8_encode(codepoint);
        } else {
            // Byte yang tidak printable: gunakan Private Use Area (U+E000 - U+E0FF)
            uint32_t codepoint = 0xE000 + n;
            BYTE_ENCODER[b] = utf8_encode(codepoint);
            n++;
        }
    }

    // Build decoder (reverse mapping)
    BYTE_DECODER.clear();
    BYTE_DECODER.reserve(256);
    for (int b = 0; b < 256; ++b) {
        BYTE_DECODER.emplace(BYTE_ENCODER[b], b);
    }
}

bool ByteLevelBPETokenizer::is_special_token(const std::string& token) const noexcept {
    return token == PAD_TOKEN || token == BOS_TOKEN || token == EOS_TOKEN || token == UNK_TOKEN;
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
                    // Karakter kontrol lainnya: gunakan \uXXXX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    escaped += buf;
                } else {
                    escaped += c;
                }
        }
    }
    return escaped;
}

void ByteLevelBPETokenizer::train(const std::string& corpus, int vocab_size) {
    if (corpus.empty() || vocab_size <= SPECIAL_TOKEN_COUNT) {
        // Minimal vocabulary dengan special tokens saja
        std::vector<std::string> specials = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN};
        vocab.clear();
        inv_vocab.clear();
        for (size_t i = 0; i < specials.size(); ++i) {
            vocab[specials[i]] = static_cast<int>(i);
            inv_vocab[static_cast<int>(i)] = specials[i];
        }
        return;
    }

    // Tokenisasi teks menjadi urutan simbol byte
    std::vector<std::vector<std::string>> words;
    std::sregex_iterator it(corpus.begin(), corpus.end(), PRETOKEN_PAT);
    std::sregex_iterator end;
    
    // Estimasi ukuran untuk reserve
    words.reserve(std::distance(it, end));
    
    for (; it != end; ++it) {
        const std::string& token = it->str();
        std::vector<std::string> syms;
        syms.reserve(token.size());
        
        for (unsigned char c : token) {
            syms.emplace_back(BYTE_ENCODER[c]);
        }
        
        if (!syms.empty()) {
            words.emplace_back(std::move(syms));
        }
    }

    if (words.empty()) return;

    // Hitung frekuensi menggunakan unordered_map
    std::unordered_map<std::vector<std::string>, int, vector_hash> freq;
    for (auto& w : words) {
        freq[std::move(w)]++;
    }

    // Inisialisasi splits
    std::unordered_map<std::vector<std::string>, std::vector<std::string>, vector_hash> splits;
    for (const auto& p : freq) {
        splits[p.first] = p.first;
    }

    // Hitung base vocabulary size
    std::set<std::string> base_vocab;
    for (int b = 0; b < 256; ++b) {
        base_vocab.insert(BYTE_ENCODER[b]);
    }

    int num_merges = std::max(0, vocab_size - static_cast<int>(base_vocab.size()) - SPECIAL_TOKEN_COUNT);
    merge_order.clear();
    merge_order.reserve(num_merges);
    merge_rank.clear();
    merge_rank.reserve(num_merges);

    for (int step = 0; step < num_merges; ++step) {
        // Hitung frekuensi pasangan menggunakan unordered_map
        std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> pair_counts;
        pair_counts.reserve(freq.size() * 10);
        
        for (const auto& wf : freq) {
            const auto& syms = splits[wf.first];
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                pair_counts[{syms[i], syms[i+1]}] += wf.second;
            }
        }

        if (pair_counts.empty()) break;

        // Cari pasangan paling sering
        auto best = std::max_element(pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        if (best->second < 2) break;

        const auto& best_pair = best->first;
        std::string merged = best_pair.first + best_pair.second;
        merge_order.emplace_back(best_pair);
        merge_rank[best_pair] = step;

        // Merge di semua kata
        for (auto& wf : freq) {
            auto& syms = splits[wf.first];
            std::vector<std::string> new_syms;
            new_syms.reserve(syms.size());
            
            for (size_t i = 0; i < syms.size(); ) {
                if (i + 1 < syms.size() && syms[i] == best_pair.first && syms[i+1] == best_pair.second) {
                    new_syms.emplace_back(merged);
                    i += 2;
                } else {
                    new_syms.emplace_back(syms[i]);
                    i++;
                }
            }
            splits[wf.first] = std::move(new_syms);
        }
    }

    // Build vocabulary
    std::vector<std::string> specials = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN};
    std::vector<std::string> all_tokens;
    all_tokens.reserve(specials.size() + base_vocab.size() + merge_order.size());
    all_tokens = specials;

    // Tambahkan base vocab
    for (const auto& s : base_vocab) {
        all_tokens.emplace_back(s);
    }

    // Tambahkan merged tokens (unik)
    std::unordered_set<std::string> merged_set;
    merged_set.reserve(merge_order.size());
    for (const auto& p : merge_order) {
        merged_set.emplace(p.first + p.second);
    }
    for (const auto& s : merged_set) {
        all_tokens.emplace_back(s);
    }

    // Sort dan hapus duplikat (kecuali special tokens)
    std::sort(all_tokens.begin() + SPECIAL_TOKEN_COUNT, all_tokens.end());
    all_tokens.erase(std::unique(all_tokens.begin() + SPECIAL_TOKEN_COUNT, all_tokens.end()), all_tokens.end());

    // Buat mappings
    vocab.clear();
    inv_vocab.clear();
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        vocab[all_tokens[i]] = static_cast<int>(i);
        inv_vocab[static_cast<int>(i)] = all_tokens[i];
    }
}

std::vector<std::string> ByteLevelBPETokenizer::apply_bpe(const std::vector<std::string>& symbols) {
    std::vector<std::string> word = symbols;
    if (word.size() <= 1) return word;

    // Implementasi yang lebih efisien seperti GPT-2
    // Gunakan priority queue untuk mencari merge terbaik
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
                // Jika menemukan rank terkecil, bisa langsung break
                // karena dalam BPE, merge yang pertama ditemukan dengan rank terkecil
                // yang akan di-merge terlebih dahulu
                if (best_rank == 0) break;
            }
        }

        if (!found) break;

        std::string merged = best_pair.first + best_pair.second;
        std::vector<std::string> new_word;
        new_word.reserve(word.size());

        for (size_t i = 0; i < word.size(); ) {
            if (i + 1 < word.size() && word[i] == best_pair.first && word[i+1] == best_pair.second) {
                new_word.emplace_back(merged);
                i += 2;
            } else {
                new_word.emplace_back(word[i]);
                i++;
            }
        }

        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    return word;
}

std::vector<int> ByteLevelBPETokenizer::encode(const std::string& text, bool add_bos, bool add_eos) {
    std::vector<int> ids;
    
    // Estimasi ukuran awal
    ids.reserve(text.size() / 2);

    if (add_bos) {
        auto it = vocab.find(BOS_TOKEN);
        if (it != vocab.end()) {
            ids.emplace_back(it->second);
        }
    }

    std::sregex_iterator it(text.begin(), text.end(), PRETOKEN_PAT);
    std::sregex_iterator end;

    for (; it != end; ++it) {
        const std::string& token = it->str();
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
        if (eos_it != vocab.end()) {
            ids.emplace_back(eos_it->second);
        }
    }

    return ids;
}

uint32_t decode_utf8_char(const std::string& str, size_t& i, int& bytes_consumed) noexcept {
    bytes_consumed = 0;
    if (i >= str.size()) return 0;

    unsigned char c = str[i];
    uint32_t codepoint = 0;
    int seq_len = 0;

    if ((c & 0x80) == 0) {
        // 1 byte
        codepoint = c;
        seq_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
        // 2 bytes - harus >= 0xC2 untuk menghindari overlong
        if (c < 0xC2) {
            bytes_consumed = 1;
            return 0xFFFD;  // Replacement character
        }
        codepoint = c & 0x1F;
        seq_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        // 3 bytes
        codepoint = c & 0x0F;
        seq_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        // 4 bytes - harus <= 0xF4 untuk code points valid
        if (c > 0xF4) {
            bytes_consumed = 1;
            return 0xFFFD;
        }
        codepoint = c & 0x07;
        seq_len = 4;
    } else {
        // Invalid UTF-8
        bytes_consumed = 1;
        return 0xFFFD;
    }

    // Baca continuation bytes
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

    // Validasi overlong encoding
    if ((seq_len == 2 && codepoint < 0x80) ||
        (seq_len == 3 && codepoint < 0x800) ||
        (seq_len == 4 && codepoint < 0x10000)) {
        bytes_consumed = seq_len;
        return 0xFFFD;
    }

    // Validasi surrogate pairs (harusnya tidak digunakan di UTF-8)
    if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
        bytes_consumed = seq_len;
        return 0xFFFD;
    }

    // Validasi maximum code point
    if (codepoint > 0x10FFFF) {
        bytes_consumed = seq_len;
        return 0xFFFD;
    }

    bytes_consumed = seq_len;
    return codepoint;
}

std::string ByteLevelBPETokenizer::decode(const std::vector<int>& ids) {
    std::vector<uint8_t> byte_sequence;
    byte_sequence.reserve(ids.size() * 2);  // Estimasi rata-rata 2 byte per token

    for (int id : ids) {
        auto it = inv_vocab.find(id);
        if (it == inv_vocab.end()) continue;

        const std::string& tok = it->second;
        if (is_special_token(tok)) continue;

        // Decode setiap karakter UTF-8 di token
        for (size_t i = 0; i < tok.size(); ) {
            int bytes_consumed = 0;
            uint32_t codepoint = decode_utf8_char(tok, i, bytes_consumed);

            if (bytes_consumed > 0) {
                // Cari mapping balik ke byte asli
                if (codepoint >= 0xE000 && codepoint <= 0xE0FF) {
                    // Private Use Area: mapping untuk non-printable bytes
                    byte_sequence.emplace_back(static_cast<uint8_t>(codepoint - 0xE000));
                } else if (codepoint < 0x80) {
                    // ASCII
                    byte_sequence.emplace_back(static_cast<uint8_t>(codepoint));
                } else if (codepoint >= 0xA0 && codepoint <= 0xFF) {
                    // Latin-1 Supplement
                    byte_sequence.emplace_back(static_cast<uint8_t>(codepoint));
                } else if (codepoint == 0xFFFD) {
                    // Invalid UTF-8, skip character
                } else {
                    // Fallback: gunakan byte decoder
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

bool ByteLevelBPETokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    // Format JSON manual dengan escape yang benar
    f << "{\n";
    f << "  \"vocab\": {\n";
    bool first = true;
    for (const auto& p : vocab) {
        if (!first) f << ",\n";
        f << "    \"" << escape_json(p.first) << "\": " << p.second;
        first = false;
    }
    f << "\n  },\n";

    f << "  \"merge_order\": [\n";
    first = true;
    for (const auto& p : merge_order) {
        if (!first) f << ",\n";
        f << "    [\"" << escape_json(p.first) << "\", \"" << escape_json(p.second) << "\"]";
        first = false;
    }
    f << "\n  ]\n";
    f << "}\n";
    
    f.close();
    return f.good();
}

bool ByteLevelBPETokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    // Gunakan library JSON yang proper jika tersedia
    // Untuk saat ini, perbaiki parsing manual dengan state machine sederhana
    std::stringstream buffer;
    buffer << f.rdbuf();
    f.close();

    std::string content = buffer.str();
    if (content.empty()) return false;

    vocab.clear();
    inv_vocab.clear();
    merge_order.clear();
    merge_rank.clear();

    // Parse vocab dengan state machine yang lebih robust
    size_t pos = 0;
    bool in_vocab = false;
    bool in_merges = false;
    int depth = 0;
    
    std::string current_key;
    std::string current_value;
    bool reading_key = false;
    bool reading_value = false;
    bool in_string = false;
    bool escape_next = false;
    
    enum State { NORMAL, VOCAB_SECTION, MERGES_SECTION };
    State state = NORMAL;
    
    // State untuk merge array
    std::string merge_first, merge_second;
    bool reading_first = false, reading_second = false;
    
    for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        
        if (escape_next) {
            if (reading_key) current_key += c;
            else if (reading_value) current_value += c;
            else if (reading_first) merge_first += c;
            else if (reading_second) merge_second += c;
            escape_next = false;
            continue;
        }
        
        if (c == '\\') {
            escape_next = true;
            continue;
        }
        
        if (c == '"') {
            if (state == NORMAL && !reading_key && !reading_value) {
                // Mencari section
                reading_key = true;
                current_key.clear();
            } else if (state == VOCAB_SECTION) {
                if (!reading_key && !reading_value) {
                    reading_key = true;
                    current_key.clear();
                } else if (reading_key) {
                    reading_key = false;
                    in_vocab = true;
                } else if (reading_value && in_string) {
                    reading_value = false;
                    in_string = false;
                    // Parse vocab entry
                    try {
                        int id = std::stoi(current_value);
                        vocab[current_key] = id;
                        inv_vocab[id] = current_key;
                    } catch (...) {}
                    current_key.clear();
                    current_value.clear();
                }
            } else if (state == MERGES_SECTION) {
                if (!reading_first && !reading_second) {
                    reading_first = true;
                    merge_first.clear();
                } else if (reading_first) {
                    reading_first = false;
                    reading_second = true;
                    merge_second.clear();
                } else if (reading_second) {
                    reading_second = false;
                    merge_order.emplace_back(merge_first, merge_second);
                    merge_rank[{merge_first, merge_second}] = merge_order.size() - 1;
                }
            }
            continue;
        }
        
        if (c == ':' && state == VOCAB_SECTION && !in_string) {
            if (!current_key.empty()) {
                reading_value = true;
                current_value.clear();
                in_string = false;
            }
            continue;
        }
        
        if ((c == ',' || c == '}' || c == ']') && !in_string) {
            if (state == VOCAB_SECTION && reading_value) {
                try {
                    int id = std::stoi(current_value);
                    vocab[current_key] = id;
                    inv_vocab[id] = current_key;
                } catch (...) {}
                current_key.clear();
                current_value.clear();
                reading_value = false;
            }
            
            if (c == '}' && state == VOCAB_SECTION) {
                state = NORMAL;
            } else if (c == ']' && state == MERGES_SECTION) {
                state = NORMAL;
            }
            continue;
        }
        
        if (reading_key) current_key += c;
        else if (reading_value) current_value += c;
        else if (reading_first) merge_first += c;
        else if (reading_second) merge_second += c;
        
        // Deteksi section
        if (state == NORMAL) {
            if (content.substr(i, 7) == "\"vocab\"") {
                state = VOCAB_SECTION;
                i += 6;
            } else if (content.substr(i, 14) == "\"merge_order\"") {
                state = MERGES_SECTION;
                i += 13;
            }
        }
    }

    return !vocab.empty();
}