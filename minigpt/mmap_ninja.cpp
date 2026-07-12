// mmap_ninja.cpp
#include "mmap_ninja.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cerrno>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>

namespace mmap_ninja {

// ============================================================
// WRITER
// ============================================================
void MMapDatasetWriter::build(const std::string& out_path,
                               const std::vector<std::vector<int>>& examples,
                               uint32_t seq_len) {
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("mmap_ninja: gagal membuka file untuk ditulis: " + out_path);
    }

    MMapHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kVersion;
    header.seq_len = seq_len;
    header.num_examples = examples.size();
    header.dtype_size = sizeof(int32_t);
    header.reserved = 0;

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!out) {
        throw std::runtime_error("mmap_ninja: gagal menulis header");
    }

    std::vector<int32_t> row(seq_len);
    for (size_t i = 0; i < examples.size(); ++i) {
        const auto& ex = examples[i];
        if (ex.size() != seq_len) {
            throw std::runtime_error(
                "mmap_ninja: example[" + std::to_string(i) + "] panjangnya " +
                std::to_string(ex.size()) + ", diharapkan seq_len=" +
                std::to_string(seq_len) + " (pad/potong dulu di sisi caller)");
        }
        for (uint32_t j = 0; j < seq_len; ++j) {
            row[j] = static_cast<int32_t>(ex[j]);
        }
        out.write(reinterpret_cast<const char*>(row.data()), (std::streamsize)(seq_len * sizeof(int32_t)));
        if (!out) {
            throw std::runtime_error("mmap_ninja: gagal menulis example[" + std::to_string(i) + "]");
        }
    }
}

// ============================================================
// READER
// ============================================================
MMapDataset::MMapDataset(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("mmap_ninja: gagal membuka file: " + path +
                                  " (" + std::strerror(errno) + ")");
    }

    struct stat st{};
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap_ninja: fstat gagal untuk " + path);
    }
    map_size_ = static_cast<size_t>(st.st_size);

    if (map_size_ < sizeof(MMapHeader)) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap_ninja: file terlalu kecil / bukan file mmap_ninja: " + path);
    }

    map_base_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_base_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        map_base_ = nullptr;
        throw std::runtime_error("mmap_ninja: mmap() gagal untuk " + path +
                                  " (" + std::strerror(errno) + ")");
    }

    const MMapHeader* header = reinterpret_cast<const MMapHeader*>(map_base_);
    if (std::memcmp(header->magic, kMagic, sizeof(kMagic)) != 0) {
        munmap(map_base_, map_size_);
        close(fd_);
        fd_ = -1;
        map_base_ = nullptr;
        throw std::runtime_error("mmap_ninja: magic number tidak cocok, file rusak/salah format: " + path);
    }
    if (header->version != kVersion) {
        munmap(map_base_, map_size_);
        close(fd_);
        fd_ = -1;
        map_base_ = nullptr;
        throw std::runtime_error("mmap_ninja: versi file tidak didukung: " + std::to_string(header->version));
    }

    // FIX: validasi dtype_size juga -- sebelumnya cuma ditulis oleh
    // writer tapi tidak pernah dicek reader. Kalau nanti ada versi
    // writer lain yang nulis tipe data berbeda (misal int16), reader
    // ini akan diam-diam salah interpretasi byte tanpa error apapun.
    if (header->dtype_size != sizeof(int32_t)) {
        munmap(map_base_, map_size_);
        close(fd_);
        fd_ = -1;
        map_base_ = nullptr;
        throw std::runtime_error("mmap_ninja: dtype_size tidak didukung (diharapkan " +
                                  std::to_string(sizeof(int32_t)) + " byte): " + path);
    }

    seq_len_ = header->seq_len;
    num_examples_ = header->num_examples;
    header_size_ = sizeof(MMapHeader);

    size_t expected_size = header_size_ + (size_t)num_examples_ * seq_len_ * sizeof(int32_t);
    if (expected_size != map_size_) {
        munmap(map_base_, map_size_);
        close(fd_);
        fd_ = -1;
        map_base_ = nullptr;
        throw std::runtime_error("mmap_ninja: ukuran file tidak konsisten dengan header (rusak?): " + path);
    }

    data_ = reinterpret_cast<const int32_t*>(
        reinterpret_cast<const char*>(map_base_) + header_size_);

    madvise(map_base_, map_size_, MADV_RANDOM);
}

MMapDataset::~MMapDataset() {
    close_mapping();
}

void MMapDataset::close_mapping() {
    if (map_base_ != nullptr) {
        munmap(map_base_, map_size_);
        map_base_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    data_ = nullptr;
}

MMapDataset::MMapDataset(MMapDataset&& other) noexcept
    : fd_(other.fd_),
      map_base_(other.map_base_),
      map_size_(other.map_size_),
      data_(other.data_),
      seq_len_(other.seq_len_),
      num_examples_(other.num_examples_),
      header_size_(other.header_size_) {
    other.fd_ = -1;
    other.map_base_ = nullptr;
    other.data_ = nullptr;
}

MMapDataset& MMapDataset::operator=(MMapDataset&& other) noexcept {
    if (this != &other) {
        close_mapping();
        fd_ = other.fd_;
        map_base_ = other.map_base_;
        map_size_ = other.map_size_;
        data_ = other.data_;
        seq_len_ = other.seq_len_;
        num_examples_ = other.num_examples_;
        header_size_ = other.header_size_;

        other.fd_ = -1;
        other.map_base_ = nullptr;
        other.data_ = nullptr;
    }
    return *this;
}

const int32_t* MMapDataset::get_example_ptr(size_t idx) const {
    if (idx >= num_examples_) {
        throw std::out_of_range("mmap_ninja: index " + std::to_string(idx) +
                                 " di luar jangkauan (" + std::to_string(num_examples_) + " examples)");
    }
    return data_ + (size_t)idx * seq_len_;
}

std::vector<int> MMapDataset::get_example(size_t idx) const {
    const int32_t* ptr = get_example_ptr(idx);
    std::vector<int> result(seq_len_);
    for (uint32_t i = 0; i < seq_len_; ++i) {
        result[i] = static_cast<int>(ptr[i]);
    }
    return result;
}

std::vector<std::vector<int>> MMapDataset::get_batch(const std::vector<size_t>& indices) const {
    std::vector<std::vector<int>> batch;
    batch.reserve(indices.size());
    for (size_t idx : indices) {
        batch.push_back(get_example(idx));
    }
    return batch;
}

void MMapDataset::prefetch(size_t idx, size_t count) const {
    if (idx >= num_examples_ || map_base_ == nullptr) return;
    size_t end = std::min(idx + count, (size_t)num_examples_);
    const char* start_ptr = reinterpret_cast<const char*>(data_ + (size_t)idx * seq_len_);
    size_t len_bytes = (end - idx) * (size_t)seq_len_ * sizeof(int32_t);

    // FIX: madvise() mensyaratkan alamat page-aligned (kelipatan page
    // size), kalau tidak gagal EINVAL secara diam-diam (return value
    // sebelumnya tidak dicek). Bulatkan alamat ke bawah ke batas
    // halaman terdekat, dan tambah panjangnya supaya range yang
    // diminta tetap tercakup penuh.
    long page_size = sysconf(_SC_PAGESIZE);
    uintptr_t addr = reinterpret_cast<uintptr_t>(start_ptr);
    uintptr_t aligned_addr = addr & ~(uintptr_t)(page_size - 1);
    size_t adjust = addr - aligned_addr;

    madvise(reinterpret_cast<void*>(aligned_addr), len_bytes + adjust, MADV_WILLNEED);
}

// ============================================================
// BATCH ITERATOR
// ============================================================
MMapBatchIterator::MMapBatchIterator(const MMapDataset& dataset, size_t batch_size,
                                      bool shuffle, unsigned seed)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), seed_(seed) {
    reset();
}

void MMapBatchIterator::reset() {
    order_.resize(dataset_.size());
    for (size_t i = 0; i < order_.size(); ++i) order_[i] = i;

    if (shuffle_) {
        std::mt19937 rng(seed_);
        std::shuffle(order_.begin(), order_.end(), rng);
        seed_ = rng();
    }
    cursor_ = 0;
}

bool MMapBatchIterator::next_batch(std::vector<std::vector<int>>& out_batch) {
    if (cursor_ >= order_.size()) {
        return false;
    }
    size_t end = std::min(cursor_ + batch_size_, order_.size());
    std::vector<size_t> indices(order_.begin() + (long)cursor_, order_.begin() + (long)end);

    out_batch = dataset_.get_batch(indices);
    cursor_ = end;
    return true;
}

size_t MMapBatchIterator::num_batches() const {
    return (dataset_.size() + batch_size_ - 1) / batch_size_;
}

} // namespace mmap_ninja