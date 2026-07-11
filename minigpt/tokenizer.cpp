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
#include <iostream>  // TAMBAHKAN untuk error handling

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

    // ASCII printable (33-126)
    for (int b = '!'; b <= '~'; ++b) printable_bytes.insert(b);
    // Latin-1 Supplement printable (161-172, 174-255)
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

    // Build decoder
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

void ByteLevelBPETokenizer::train(const std::string& corpus, int vocab_size) {
    // VALIDASI INPUT
    if (corpus.empty()) {
        std::cerr << "⚠️ Warning: Corpus kosong, menggunakan vocab minimal" << std::endl;
        // Buat vocab minimal
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

    // Tokenisasi dengan regex iterator
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

    // ... sisa implementasi BPE training sama seperti sebelumnya ...
    // (kode BPE training yang sudah ada)
}

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

// ... sisa implementasi save/load dan apply_bpe sama seperti sebelumnya ...