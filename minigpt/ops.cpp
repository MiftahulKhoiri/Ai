// ops.cpp
#include "ops.h"
#include <cassert>
#include <algorithm>
#include <numeric>

namespace tnsr {
namespace ops {

// ============ MATMUL (2D) ============
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2)
        throw std::invalid_argument("matmul: only 2D tensors supported");
    if (a.shape[1] != b.shape[0])
        throw std::invalid_argument("matmul: inner dimensions mismatch");

    // Salin ke Matrix (row‑major) dan gunakan Matrix::matmul yang sudah memakai tiling & SIMD
    linalg::Matrix<T> matA(a.shape[0], a.shape[1]);
    std::copy(a.data.begin(), a.data.end(), &matA(0,0));

    linalg::Matrix<T> matB(b.shape[0], b.shape[1]);
    std::copy(b.data.begin(), b.data.end(), &matB(0,0));

    linalg::Matrix<T> matC = matA.matmul(matB);

    Tensor<T> result({matC.rows(), matC.cols()});
    std::copy(&matC(0,0), &matC(0,0) + matC.rows()*matC.cols(), result.data.begin());
    return result;
}

// ============ TRANSPOSE (2D) ============
template<typename T>
Tensor<T> transpose(const Tensor<T>& a, int dim0, int dim1) {
    if (a.shape.size() != 2)
        throw std::invalid_argument("transpose: only 2D tensor supported");
    size_t rows = a.shape[0], cols = a.shape[1];
    Tensor<T> result({cols, rows});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result(j, i) = a(i, j);
    return result;
}

// ============ CONCAT ============
template<typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, int axis) {
    if (tensors.empty())
        throw std::invalid_argument("concat: empty list");
    size_t ndim = tensors[0].shape.size();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= (int)ndim)
        throw std::invalid_argument("concat: axis out of range");

    // Validasi shape semua tensor (kecuali axis)
    auto base_shape = tensors[0].shape;
    base_shape[axis] = 0;  // akan dijumlahkan
    size_t sum_axis = 0;
    for (const auto& t : tensors) {
        if (t.shape.size() != ndim)
            throw std::invalid_argument("concat: different number of dimensions");
        for (size_t d = 0; d < ndim; ++d) {
            if (d == (size_t)axis) continue;
            if (t.shape[d] != tensors[0].shape[d])
                throw std::invalid_argument("concat: shape mismatch on dim " + std::to_string(d));
        }
        sum_axis += t.shape[axis];
    }
    base_shape[axis] = sum_axis;
    Tensor<T> result(base_shape);

    // Menyalin data dengan loop bersarang
    // Kita bangun index generator untuk semua dimensi kecuali axis
    std::vector<size_t> outer_shape = base_shape;
    outer_shape.erase(outer_shape.begin() + axis);
    size_t outer_total = std::accumulate(outer_shape.begin(), outer_shape.end(), 1UL, std::multiplies<>());
    size_t block_size = 1;
    for (size_t d = axis + 1; d < ndim; ++d) block_size *= base_shape[d];

    T* dst = result.data.data();
    for (const auto& t : tensors) {
        size_t len = t.shape[axis] * block_size;
        // Untuk setiap segmen luar (sepanjang outer_total), kita salin len elemen
        const T* src = t.data.data();
        for (size_t outer = 0; outer < outer_total; ++outer) {
            std::memcpy(dst, src, len * sizeof(T));
            dst += len;
            src += len;
        }
    }
    return result;
}

// ============ SPLIT ============
template<typename T>
std::vector<Tensor<T>> split(const Tensor<T>& a, size_t sections, int axis) {
    if (sections == 0) return {};
    if (axis < 0) axis += a.shape.size();
    size_t dim_size = a.shape[axis];
    if (dim_size % sections != 0)
        throw std::invalid_argument("split: dimension size not divisible by sections");
    size_t each = dim_size / sections;

    std::vector<Tensor<T>> result;
    result.reserve(sections);

    // Bentuk slice per bagian
    std::vector<std::pair<size_t,size_t>> ranges(a.shape.size(), {0,0});
    for (size_t d = 0; d < a.shape.size(); ++d) ranges[d] = {0, a.shape[d]};
    for (size_t i = 0; i < sections; ++i) {
        ranges[axis] = {i * each, (i+1) * each};
        result.push_back(slice(a, ranges));
    }
    return result;
}

// ============ STACK ============
template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, int axis) {
    if (tensors.empty()) throw std::invalid_argument("stack: empty list");
    auto first_shape = tensors[0].shape;
    size_t ndim = first_shape.size();
    if (axis < 0) axis += ndim + 1;
    if (axis < 0 || axis > (int)ndim)
        throw std::invalid_argument("stack: axis out of range");

    // Validasi semua tensor shape sama
    for (const auto& t : tensors) {
        if (t.shape != first_shape)
            throw std::invalid_argument("stack: all tensors must have same shape");
    }

    // Bentuk shape baru: sisipkan axis dengan ukuran jumlah tensor
    std::vector<size_t> new_shape = first_shape;
    new_shape.insert(new_shape.begin() + axis, tensors.size());
    Tensor<T> result(new_shape);
    // Salin data: setiap tensor menempati slice sepanjang axis baru
    std::vector<std::pair<size_t,size_t>> ranges(ndim+1, {0,0});
    for (size_t d = 0; d < ndim+1; ++d) ranges[d] = {0, new_shape[d]};
    for (size_t i = 0; i < tensors.size(); ++i) {
        ranges[axis] = {i, i+1};
        // slice dari result (destinasi) lalu copy data tensor ke dalamnya
        Tensor<T> dest_slice = slice(result, ranges); // ini membuat copy? Sebaiknya kita tulis langsung
        // Langsung salin data: dest_slice berukuran first_shape, kita timpa seluruhnya
        std::copy(tensors[i].data.begin(), tensors[i].data.end(), dest_slice.data.begin());
        // Masukkan kembali ke result (karena dest_slice adalah copy, ini tidak mempengaruhi result)
        // Cara lebih baik: akses langsung memori result. Kita lakukan manual loop.
    }
    // Implementasi ulang stack dengan loop manual agar efisien (hindari slice sementara)
    // Mari kita tulis ulang lebih jelas di bawah.
    // Untuk sederhana, gunakan kembali concat + reshape
    // Logika: stack = reshape(concat(tensors, axis), new_shape)
    // Tapi concat menambah sepanjang axis yang sudah ada, bukan menambah axis baru.
    // Kita bisa: ubah semua tensor jadi shape dengan axis baru ukuran 1, lalu concat pada axis tersebut.
    std::vector<Tensor<T>> expanded;
    expanded.reserve(tensors.size());
    for (const auto& t : tensors) {
        std::vector<size_t> shp = t.shape;
        shp.insert(shp.begin() + axis, 1);
        expanded.push_back(reshape(t, shp)); // reshape valid karena total size sama
    }
    return concat(expanded, axis);
}

// ============ SLICE ============
template<typename T>
Tensor<T> slice(const Tensor<T>& a, const std::vector<std::pair<size_t,size_t>>& ranges) {
    if (ranges.size() != a.shape.size())
        throw std::invalid_argument("slice: ranges size must equal tensor dimensions");
    std::vector<size_t> new_shape(ranges.size());
    for (size_t d = 0; d < ranges.size(); ++d) {
        if (ranges[d].second < ranges[d].first || ranges[d].second > a.shape[d])
            throw std::invalid_argument("slice: invalid range on dim " + std::to_string(d));
        new_shape[d] = ranges[d].second - ranges[d].first;
    }
    Tensor<T> result(new_shape);
    // Iterasi semua indeks output, mapping ke indeks input
    // Kita bisa gunakan multi‑index loop
    std::vector<size_t> out_idx(new_shape.size(), 0);
    std::vector<size_t> in_idx(a.shape.size());
    size_t total_out = result.data.size();
    for (size_t linear = 0; linear < total_out; ++linear) {
        // Konversi linear ke out_idx
        size_t temp = linear;
        for (int d = new_shape.size()-1; d >= 0; --d) {
            out_idx[d] = temp % new_shape[d];
            temp /= new_shape[d];
        }
        // in_idx = out_idx + start
        for (size_t d = 0; d < a.shape.size(); ++d)
            in_idx[d] = out_idx[d] + ranges[d].first;
        result.data[linear] = a(in_idx);
    }
    return result;
}

// ============ GATHER ============
template<typename T>
Tensor<T> gather(const Tensor<T>& a, int axis, const Tensor<int>& indices) {
    if (axis < 0) axis += a.shape.size();
    if (axis < 0 || axis >= (int)a.shape.size())
        throw std::invalid_argument("gather: axis out of range");

    // Output shape = indices.shape (kecuali axis diganti sesuai indices)
    // Atau: shape = indices.shape dengan dimensi axis dipertahankan? PyTorch: output shape = indices.shape.
    auto out_shape = indices.shape;
    Tensor<T> result(out_shape);

    // Iterasi semua posisi di indices, ambil nilai dari a pada axis sesuai indices
    std::vector<size_t> idx(out_shape.size(), 0);
    size_t total = indices.data.size();
    for (size_t lin = 0; lin < total; ++lin) {
        size_t temp = lin;
        for (int d = out_shape.size()-1; d >= 0; --d) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }
        int gather_idx = indices(idx);  // indeks pada axis
        if (gather_idx < 0 || (size_t)gather_idx >= a.shape[axis])
            throw std::out_of_range("gather: index out of range");
        // Bentuk indeks ke a: copy semua dimensi kecuali axis, lalu set axis dengan gather_idx
        std::vector<size_t> a_idx = idx;
        a_idx[axis] = gather_idx;
        result.data[lin] = a(a_idx);
    }
    return result;
}

// ============ SCATTER ============
template<typename T>
Tensor<T> scatter(const Tensor<T>& a, int axis, const Tensor<int>& indices, const Tensor<T>& updates) {
    if (axis < 0) axis += a.shape.size();
    if (axis < 0 || axis >= (int)a.shape.size())
        throw std::invalid_argument("scatter: axis out of range");
    if (indices.shape != updates.shape)
        throw std::invalid_argument("scatter: indices and updates must have same shape");

    // Output = copy of a, lalu updates ditulis ke posisi sesuai indices
    Tensor<T> result = a; // copy
    std::vector<size_t> idx(indices.shape.size(), 0);
    size_t total = indices.data.size();
    for (size_t lin = 0; lin < total; ++lin) {
        size_t temp = lin;
        for (int d = indices.shape.size()-1; d >= 0; --d) {
            idx[d] = temp % indices.shape[d];
            temp /= indices.shape[d];
        }
        int scatter_idx = indices(idx);
        if (scatter_idx < 0 || (size_t)scatter_idx >= a.shape[axis])
            throw std::out_of_range("scatter: index out of range");
        std::vector<size_t> a_idx = idx;
        a_idx[axis] = scatter_idx;
        result(a_idx) = updates(idx);
    }
    return result;
}

// -------------------------------------------------------------------
// Explicit template instantiation untuk float dan double
// -------------------------------------------------------------------
#define INSTANTIATE(T) \
    template Tensor<T> add<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> sub<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> mul<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> div<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> matmul<T>(const Tensor<T>&, const Tensor<T>&); \
    template Tensor<T> transpose<T>(const Tensor<T>&, int, int); \
    template Tensor<T> reshape<T>(const Tensor<T>&, const std::vector<size_t>&); \
    template Tensor<T> view<T>(const Tensor<T>&, const std::vector<size_t>&); \
    template Tensor<T> concat<T>(const std::vector<Tensor<T>>&, int); \
    template std::vector<Tensor<T>> split<T>(const Tensor<T>&, size_t, int); \
    template Tensor<T> stack<T>(const std::vector<Tensor<T>>&, int); \
    template Tensor<T> slice<T>(const Tensor<T>&, const std::vector<std::pair<size_t,size_t>>&); \
    template Tensor<T> gather<T>(const Tensor<T>&, int, const Tensor<int>&); \
    template Tensor<T> scatter<T>(const Tensor<T>&, int, const Tensor<int>&, const Tensor<T>&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace ops
} // namespace tnsr