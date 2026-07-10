#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <utility>

struct pair_hash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept;
};

struct vector_hash {
    std::size_t operator()(const std::vector<std::string>& v) const noexcept;
};

class ByteLevelBPETokenizer {
public:
    ByteLevelBPETokenizer();
    
    void train(const std::string& corpus, int vocab_size);
    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false);
    std::string decode(const std::vector<int>& ids);
    
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    
private:
    static bool byte_maps_initialized;
    static std::array<std::string, 256> BYTE_ENCODER;
    static std::unordered_map<std::string, int> BYTE_DECODER;
    
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    
    std::vector<std::pair<std::string, std::string>> merge_order;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> merge_rank;
    
    static void init_byte_maps();
    static std::string utf8_encode(uint32_t codepoint) noexcept;
    
    bool is_special_token(const std::string& token) const noexcept;
    std::string escape_json(const std::string& str) const;
    std::vector<std::string> apply_bpe(const std::vector<std::string>& symbols);
    
    // Constants
    static constexpr const char* PAD_TOKEN = "<pad>";
    static constexpr const char* BOS_TOKEN = "<bos>";
    static constexpr const char* EOS_TOKEN = "<eos>";
    static constexpr const char* UNK_TOKEN = "<unk>";
    static constexpr int SPECIAL_TOKEN_COUNT = 4;
};

uint32_t decode_utf8_char(const std::string& str, size_t& i, int& bytes_consumed) noexcept;

#endif // TOKENIZER_H