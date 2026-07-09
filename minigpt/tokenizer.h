#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

struct ByteLevelBPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::vector<std::pair<std::string, std::string>> merge_order;
    std::map<std::pair<std::string, std::string>, int> merge_rank; // pakai std::map agar tidak perlu hash

    void train(const std::string& corpus, int vocab_size = 400);
    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false);
    std::string decode(const std::vector<int>& ids);
    void save(const std::string& path);
    void load(const std::string& path);

private:
    std::vector<std::string> apply_bpe(const std::vector<std::string>& symbols);
    void init_byte_maps();
};