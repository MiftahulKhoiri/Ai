#include "tokenizer.h"
#include <regex>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <stdexcept>

// Inisialisasi static members
bool ByteLevelBPETokenizer::byte_maps_initialized = false;
std::unordered_map<int, std::string> ByteLevelBPETokenizer::BYTE_ENCODER;
std::unordered_map<std::string, int> ByteLevelBPETokenizer::BYTE_DECODER;

static const std::regex PRETOKEN_PAT(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+)"
);

ByteLevelBPETokenizer::ByteLevelBPETokenizer() {
    init_byte_maps();
}

std::string ByteLevelBPETokenizer::utf8_encode(uint32_t codepoint) {
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
    // ASCII printable (33-126)
    for (int b = '!'; b <= '~'; ++b) printable_bytes.push_back(b);
    // Latin-1 Supplement printable (161-172, 174-255)
    for (int b = 0xA1; b <= 0xAC; ++b) printable_bytes.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) printable_bytes.push_back(b);
    
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(printable_bytes.begin(), printable_bytes.end(), b) != printable_bytes.end()) {
            // Byte yang printable: gunakan langsung sebagai karakter Latin-1
            // Encode ke UTF-8 (Latin-1 code points 0xA0-0xFF menjadi UTF-8 2-byte)
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
    for (auto& p : BYTE_ENCODER) {
        BYTE_DECODER[p.second] = p.first;
    }
}

bool ByteLevelBPETokenizer::is_special_token(const std::string& token) const {
    return token == "<pad>" || token == "<bos>" || token == "<eos>" || token == "<unk>";
}

void ByteLevelBPETokenizer::train(const std::string& corpus, int vocab_size) {
    // Tokenisasi teks menjadi urutan simbol byte
    std::vector<std::vector<std::string>> words;
    std::sregex_iterator it(corpus.begin(), corpus.end(), PRETOKEN_PAT);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        std::string token = it->str();
        std::vector<std::string> syms;
        for (unsigned char c : token) {
            auto enc_it = BYTE_ENCODER.find(c);
            if (enc_it != BYTE_ENCODER.end()) {
                syms.push_back(enc_it->second);
            }
        }
        if (!syms.empty()) {
            words.push_back(syms);
        }
    }
    
    if (words.empty()) return;
    
    // Hitung frekuensi
    std::map<std::vector<std::string>, int> freq;
    for (auto& w : words) freq[w]++;
    
    // Inisialisasi splits
    std::map<std::vector<std::string>, std::vector<std::string>> splits;
    for (auto& p : freq) splits[p.first] = p.first;
    
    // Hitung base vocabulary size
    std::set<std::string> base_vocab;
    for (auto& p : BYTE_ENCODER) base_vocab.insert(p.second);
    
    int num_merges = std::max(0, vocab_size - (int)base_vocab.size() - 4);  // 4 special tokens
    merge_order.clear();
    merge_rank.clear();
    
    for (int step = 0; step < num_merges; ++step) {
        // Hitung frekuensi pasangan
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
        
        if (best->second < 2) break;
        
        std::pair<std::string, std::string> best_pair = best->first;
        std::string merged = best_pair.first + best_pair.second;
        merge_order.push_back(best_pair);
        merge_rank[best_pair] = step;
        
        // Merge di semua kata
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
    
    // Tambahkan merged tokens (unik)
    std::set<std::string> merged_set;
    for (auto& p : merge_order) {
        merged_set.insert(p.first + p.second);
    }
    for (auto& s : merged_set) all_tokens.push_back(s);
    
    // Sort dan hapus duplikat (kecuali special tokens)
    std::sort(all_tokens.begin() + 4, all_tokens.end());
    all_tokens.erase(std::unique(all_tokens.begin() + 4, all_tokens.end()), all_tokens.end());
    
    // Buat mappings
    vocab.clear();
    inv_vocab.clear();
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        vocab[all_tokens[i]] = static_cast<int>(i);
        inv_vocab[i] = all_tokens[i];
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
        
        for (size_t i = 0; i < word.size(); ) {
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
        
        for (unsigned char c : token) {
            auto enc_it = BYTE_ENCODER.find(c);
            if (enc_it != BYTE_ENCODER.end()) {
                syms.push_back(enc_it->second);
            }
        }
        
        auto bpe_result = apply_bpe(syms);
        for (auto& s : bpe_result) {
            auto vit = vocab.find(s);
            if (vit != vocab.end()) {
                ids.push_back(vit->second);
            } else {
                auto unk_it = vocab.find("<unk>");
                if (unk_it != vocab.end()) {
                    ids.push_back(unk_it->second);
                }
            }
        }
    }
    
    if (add_eos) {
        auto eos_it = vocab.find("<eos>");
        if (eos_it != vocab.end()) ids.push_back(eos_it->second);
    }
    
    return ids;
}

uint32_t decode_utf8_char(const std::string& str, size_t& i, int& bytes_consumed) {
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
        // 2 bytes
        codepoint = c & 0x1F;
        seq_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        // 3 bytes
        codepoint = c & 0x0F;
        seq_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        // 4 bytes
        codepoint = c & 0x07;
        seq_len = 4;
    } else {
        // Invalid UTF-8
        bytes_consumed = 1;
        return 0;
    }
    
    // Baca continuation bytes
    for (int j = 1; j < seq_len; j++) {
        if (i + j >= str.size()) {
            bytes_consumed = j;
            return 0;
        }
        unsigned char cont = str[i + j];
        if ((cont & 0xC0) != 0x80) {
            bytes_consumed = j;
            return 0;
        }
        codepoint = (codepoint << 6) | (cont & 0x3F);
    }
    
    bytes_consumed = seq_len;
    return codepoint;
}

std::string ByteLevelBPETokenizer::decode(const std::vector<int>& ids) {
    std::vector<uint8_t> byte_sequence;
    
    for (int id : ids) {
        auto it = inv_vocab.find(id);
        if (it == inv_vocab.end()) continue;
        
        std::string tok = it->second;
        if (is_special_token(tok)) continue;
        
        // Decode setiap karakter UTF-8 di token
        for (size_t i = 0; i < tok.size(); ) {
            int bytes_consumed = 0;
            uint32_t codepoint = decode_utf8_char(tok, i, bytes_consumed);
            
            if (bytes_consumed > 0) {
                // Cari mapping balik ke byte asli
                if (codepoint >= 0xE000 && codepoint <= 0xE0FF) {
                    // Private Use Area: mapping untuk non-printable bytes
                    byte_sequence.push_back(static_cast<uint8_t>(codepoint - 0xE000));
                } else if (codepoint < 0x80) {
                    // ASCII
                    byte_sequence.push_back(static_cast<uint8_t>(codepoint));
                } else if (codepoint >= 0xA0 && codepoint <= 0xFF) {
                    // Latin-1 Supplement
                    byte_sequence.push_back(static_cast<uint8_t>(codepoint));
                } else {
                    // Fallback: gunakan byte decoder
                    std::string char_str = tok.substr(i, bytes_consumed);
                    auto dec_it = BYTE_DECODER.find(char_str);
                    if (dec_it != BYTE_DECODER.end()) {
                        byte_sequence.push_back(static_cast<uint8_t>(dec_it->second));
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

void ByteLevelBPETokenizer::save(const std::string& path) {
    // Simpan sebagai JSON menggunakan Python untuk kemudahan
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Format JSON manual yang sederhana
    f << "{\n";
    f << "  \"vocab\": {\n";
    bool first = true;
    for (auto& p : vocab) {
        if (!first) f << ",\n";
        f << "    \"" << p.first << "\": " << p.second;
        first = false;
    }
    f << "\n  },\n";
    
    f << "  \"merge_order\": [\n";
    first = true;
    for (auto& p : merge_order) {
        if (!first) f << ",\n";
        f << "    [\"" << p.first << "\", \"" << p.second << "\"]";
        first = false;
    }
    f << "\n  ]\n";
    f << "}\n";
    f.close();
}

void ByteLevelBPETokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::stringstream buffer;
    buffer << f.rdbuf();
    f.close();
    
    std::string content = buffer.str();
    
    // Parse JSON secara manual (sederhana)
    // Ini hanya implementasi dasar, untuk produksi gunakan library JSON
    
    vocab.clear();
    inv_vocab.clear();
    merge_order.clear();
    merge_rank.clear();
    
    // Parse vocab
    size_t pos = content.find("\"vocab\"");
    if (pos != std::string::npos) {
        pos = content.find("{", pos);
        size_t end_pos = content.find("}", pos);
        std::string vocab_str = content.substr(pos + 1, end_pos - pos - 1);
        
        // Parse setiap baris "token": id
        size_t line_start = 0;
        while (line_start < vocab_str.length()) {
            size_t key_start = vocab_str.find("\"", line_start);
            if (key_start == std::string::npos) break;
            size_t key_end = vocab_str.find("\"", key_start + 1);
            std::string token = vocab_str.substr(key_start + 1, key_end - key_start - 1);
            
            size_t colon = vocab_str.find(":", key_end);
            size_t comma = vocab_str.find(",", colon);
            if (comma == std::string::npos) comma = vocab_str.length();
            
            std::string id_str = vocab_str.substr(colon + 1, comma - colon - 1);
            // Trim whitespace
            id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
            id_str.erase(id_str.find_last_not_of(" \t\n\r") + 1);
            
            int id = std::stoi(id_str);
            vocab[token] = id;
            inv_vocab[id] = token;
            
            line_start = comma + 1;
        }
    }
    
    // Parse merge_order
    pos = content.find("\"merge_order\"");
    if (pos != std::string::npos) {
        pos = content.find("[", pos);
        size_t end_pos = content.find("]", pos);
        std::string merge_str = content.substr(pos + 1, end_pos - pos - 1);
        
        size_t pair_start = 0;
        while (pair_start < merge_str.length()) {
            size_t first_start = merge_str.find("\"", pair_start);
            if (first_start == std::string::npos) break;
            size_t first_end = merge_str.find("\"", first_start + 1);
            std::string first = merge_str.substr(first_start + 1, first_end - first_start - 1);
            
            size_t second_start = merge_str.find("\"", first_end + 1);
            size_t second_end = merge_str.find("\"", second_start + 1);
            std::string second = merge_str.substr(second_start + 1, second_end - second_start - 1);
            
            merge_order.push_back({first, second});
            merge_rank[{first, second}] = merge_order.size() - 1;
            
            size_t bracket = merge_str.find("]", second_end);
            pair_start = bracket + 2;  // Skip past "],"
            if (pair_start >= merge_str.length()) break;
        }
    }
}