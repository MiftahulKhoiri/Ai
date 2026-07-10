#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class ByteLevelBPETokenizer {
public:
    ByteLevelBPETokenizer();

    // Vocabulary
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;

    // BPE merges
    std::vector<std::pair<std::string, std::string>> merge_order;
    std::map<std::pair<std::string, std::string>, int> merge_rank;

    // Training
    void train(const std::string& corpus, std::size_t vocab_size = 400);

    // Encode / Decode
    std::vector<int> encode(
        const std::string& text,
        bool add_bos = false,
        bool add_eos = false
    );

    std::string decode(const std::vector<int>& ids) const;

    // Save / Load
    void save(const std::string& path) const;
    void load(const std::string& path);

    [[nodiscard]] inline std::size_t vocab_size() const noexcept {
        return vocab.size();
    }

private:
    // Byte encoder / decoder
    static bool byte_maps_initialized;
    static std::unordered_map<int, std::string> BYTE_ENCODER;
    static std::unordered_map<std::string, int> BYTE_DECODER;

    void init_byte_maps();

    // UTF-8 helper
    static std::string utf8_encode(uint32_t codepoint);

    // BPE
    std::vector<std::string> apply_bpe(
        const std::vector<std::string>& symbols
    );

    bool is_special_token(const std::string& token) const;
};