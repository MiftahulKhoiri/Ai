// mmap_ninja.h
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mmap_ninja {

constexpr char kMagic[8] = {'M', 'M', 'P', 'N', 'I', 'N', 'J', 'A'};
constexpr uint32_t kVersion = 1;

#pragma pack(push, 1)
struct MMapHeader {
    char magic[8];
    uint32_t version;
    uint32_t seq_len;
    uint64_t num_examples;
    uint32_t dtype_size;
    uint32_t reserved;
};
#pragma pack(pop)

// ============================================================
// WRITER — konversi data (di RAM) menjadi file biner mmap_ninja.
// Dipakai SEKALI di awal (offline), bukan saat training.
// ============================================================
class MMapDatasetWriter {
public:
    static void build(const std::string& out_path,
                       const std::vector<std::vector<int>>& examples,
                       uint32_t seq_len);
};

// ============================================================
// READER — mmap file biner, baca via demand paging.
// ============================================================
class MMapDataset {
public:
    explicit MMapDataset(const std::string& path);
    ~MMapDataset();

    MMapDataset(const MMapDataset&) = delete;
    MMapDataset& operator=(const MMapDataset&) = delete;
    MMapDataset(MMapDataset&& other) noexcept;
    MMapDataset& operator=(MMapDataset&& other) noexcept;

    size_t size() const noexcept { return static_cast<size_t>(num_examples_); }
    uint32_t seq_len() const noexcept { return seq_len_; }

    // Pointer langsung ke virtual memory (tidak menyalin apa pun).
    // Valid selama objek MMapDataset masih hidup.
    const int32_t* get_example_ptr(size_t idx) const;

    // Menyalin SATU contoh ke std::vector<int> — dipakai untuk API
    // model.forward() yang butuh std::vector<int> biasa.
    std::vector<int> get_example(size_t idx) const;

    // Menyalin BEBERAPA contoh (batch). Hanya index yang diminta yang
    // benar-benar disalin ke heap RAM; sisanya tetap di virtual
    // memory / page cache OS, tidak pernah dimuat penuh.
    std::vector<std::vector<int>> get_batch(const std::vector<size_t>& indices) const;

    // Hint opsional ke kernel Linux: "aku akan butuh range ini sebentar
    // lagi" — supaya OS mulai baca dari disk lebih awal (readahead).
    void prefetch(size_t idx, size_t count = 1) const;

private:
    void close_mapping();

    int fd_ = -1;
    void* map_base_ = nullptr;
    size_t map_size_ = 0;
    const int32_t* data_ = nullptr;
    uint32_t seq_len_ = 0;
    uint64_t num_examples_ = 0;
    size_t header_size_ = 0;
};

// ============================================================
// BATCH ITERATOR — shuffle + iterasi batch tanpa memuat seluruh
// dataset ke RAM. Yang disimpan cuma array index (jauh lebih kecil
// dari data aslinya), bukan datanya sendiri.
// ============================================================
class MMapBatchIterator {
public:
    MMapBatchIterator(const MMapDataset& dataset, size_t batch_size,
                       bool shuffle = true, unsigned seed = 42);

    void reset();
    bool next_batch(std::vector<std::vector<int>>& out_batch);
    size_t num_batches() const;

private:
    const MMapDataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    unsigned seed_;
    std::vector<size_t> order_;
    size_t cursor_ = 0;
};

} // namespace mmap_ninja