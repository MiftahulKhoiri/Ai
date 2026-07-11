// tokenizer.h
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <utility>
#include <cstdint>

struct pair_hash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept;
};

struct vector_hash {
    std::size_t operator()(const std::vector<std::string>& v) const noexcept;
};

class ByteLevelBPETokenizer {
public:
    ByteLevelBPETokenizer();

    // ============================================================
    // CORE METHODS
    // ============================================================
    void train(const std::string& corpus, int vocab_size);
    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false);
    std::string decode(const std::vector<int>& ids);

    // ============================================================
    // SAVE / LOAD
    // ============================================================
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // ============================================================
    // GETTERS
    // ============================================================
    const std::unordered_map<std::string, int>& get_vocab() const { return vocab; }
    const std::unordered_map<int, std::string>& get_inv_vocab() const { return inv_vocab; }
    const std::vector<std::pair<std::string, std::string>>& get_merge_order() const { return merge_order; }
    
    int vocab_size() const { return static_cast<int>(vocab.size()); }
    
    // ============================================================
    // SPECIAL TOKEN IDs
    // ============================================================
    int get_eos_token_id() const {
        auto it = vocab.find(EOS_TOKEN);
        return (it != vocab.end()) ? it->second : -1;
    }
    
    int get_bos_token_id() const {
        auto it = vocab.find(BOS_TOKEN);
        return (it != vocab.end()) ? it->second : -1;
    }
    
    int get_pad_token_id() const {
        auto it = vocab.find(PAD_TOKEN);
        return (it != vocab.end()) ? it->second : -1;
    }
    
    int get_unk_token_id() const {
        auto it = vocab.find(UNK_TOKEN);
        return (it != vocab.end()) ? it->second : -1;
    }

    // ============================================================
    // SETTERS (untuk deserialisasi)
    // ============================================================
    void set_vocab(const std::unordered_map<std::string, int>& v) { vocab = v; }
    void set_inv_vocab(const std::unordered_map<int, std::string>& v) { inv_vocab = v; }
    void set_merge_order(const std::vector<std::pair<std::string, std::string>>& m) { merge_order = m; }

private:
    // ============================================================
    // STATIC MEMBERS
    // ============================================================
    static bool byte_maps_initialized;
    static std::array<std::string, 256> BYTE_ENCODER;
    static std::unordered_map<std::string, int> BYTE_DECODER;

    // ============================================================
    // MEMBER VARIABLES
    // ============================================================
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::vector<std::pair<std::string, std::string>> merge_order;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> merge_rank;

    // ============================================================
    // PRIVATE METHODS
    // ============================================================
    static void init_byte_maps();
    static std::string utf8_encode(uint32_t codepoint) noexcept;

    bool is_special_token(const std::string& token) const noexcept;
    std::string escape_json(const std::string& str) const;
    std::vector<std::string> apply_bpe(const std::vector<std::string>& symbols);

    // ============================================================
    // CONSTANTS
    // ============================================================
    static constexpr const char* PAD_TOKEN = "<pad>";
    static constexpr const char* BOS_TOKEN = "<bos>";
    static constexpr const char* EOS_TOKEN = "<eos>";
    static constexpr const char* UNK_TOKEN = "<unk>";
    static constexpr int SPECIAL_TOKEN_COUNT = 4;
};

// ============================================================
// HELPER FUNCTIONS
// ============================================================
uint32_t decode_utf8_char(const std::string& str, size_t& i, int& bytes_consumed) noexcept;

#endif // TOKENIZER_H