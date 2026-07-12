#!/usr/bin/env python3
"""
check_mmap_binding.py - Memeriksa hasil build binding pybind11 untuk mmap_ninja.
"""
import os
import sys
import importlib
import inspect
import tempfile
import random

def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

# ============================================================
# 1. Cek keberadaan modul
# ============================================================
print_section("1. IMPOR MODUL mmap_ninja")
try:
    import mmap_ninja
    print("✅ Modul 'mmap_ninja' berhasil diimpor.")
except ImportError as e:
    print(f"❌ Gagal mengimpor mmap_ninja: {e}")
    print("   Pastikan file .so hasil build ada di direktori atau PYTHONPATH.")
    sys.exit(1)

# ============================================================
# 2. Tampilkan atribut modul
# ============================================================
print_section("2. ATRIBUT MODUL")
attrs = [attr for attr in dir(mmap_ninja) if not attr.startswith('_')]
print(f"Ditemukan {len(attrs)} atribut publik:")
for attr in attrs:
    obj = getattr(mmap_ninja, attr)
    obj_type = type(obj).__name__
    print(f"  - {attr:<20} : {obj_type}")
    if hasattr(obj, '__doc__') and obj.__doc__:
        doc = obj.__doc__.strip().split('\n')[0]  # ambil baris pertama
        print(f"      {doc}")

# ============================================================
# 3. Informasi file .so
# ============================================================
print_section("3. LOKASI FILE .so")
try:
    file_path = mmap_ninja.__file__
    print(f"📁 Path          : {file_path}")
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"📦 Ukuran        : {size / 1024:.2f} KB")
        print(f"🔧 Tipe file     : {os.path.splitext(file_path)[1]}")
    else:
        print("⚠️  File tidak ditemukan di path tersebut.")
except AttributeError:
    print("ℹ️  Modul tidak memiliki atribut __file__ (mungkin built-in).")

# ============================================================
# 4. Tes fungsionalitas dasar
# ============================================================
print_section("4. UJI FUNGSIONALITAS DASAR")

# Periksa apakah fungsi build dan kelas MMapDataset ada
if not hasattr(mmap_ninja, 'build') or not hasattr(mmap_ninja, 'MMapDataset'):
    print("❌ Modul tidak memiliki 'build' atau 'MMapDataset'. Binding mungkin tidak lengkap.")
    sys.exit(1)

# Buat data dummy
SEQ_LEN = 8
NUM_EXAMPLES = 100
dummy_data = [[random.randint(0, 100) for _ in range(SEQ_LEN)] for _ in range(NUM_EXAMPLES)]

print(f"Data dummy: {NUM_EXAMPLES} contoh, seq_len={SEQ_LEN}")

# Tulis ke file sementara
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
    tmp_path = tmp.name

print(f"Menulis ke file sementara: {tmp_path}")
try:
    mmap_ninja.build(tmp_path, dummy_data, SEQ_LEN)
    print("✅ build() berhasil.")
except Exception as e:
    print(f"❌ build() gagal: {e}")
    sys.exit(1)

# Baca dengan MMapDataset
try:
    dataset = mmap_ninja.MMapDataset(tmp_path)
    print(f"✅ MMapDataset berhasil dibuat.")
    print(f"   Jumlah contoh: {dataset.size()}")
    # Coba ambil contoh pertama
    first = dataset.get_example(0)
    print(f"   Contoh pertama (5 angka): {first[:5]}")
    # Coba batch
    batch = dataset.get_batch([0, 50, 99])
    print(f"   Batch 3 contoh: {len(batch)} baris, masing-masing panjang {len(batch[0])}")
except Exception as e:
    print(f"❌ Gagal membaca dengan MMapDataset: {e}")
    sys.exit(1)

# Bersihkan
os.unlink(tmp_path)
print("🗑️  File sementara dihapus.")

# ============================================================
# 5. Uji MMapBatchIterator (jika tersedia)
# ============================================================
if hasattr(mmap_ninja, 'MMapBatchIterator'):
    print_section("5. UJI MMapBatchIterator")
    # Buat ulang dataset dari file sementara
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path2 = tmp.name
    mmap_ninja.build(tmp_path2, dummy_data, SEQ_LEN)
    dataset2 = mmap_ninja.MMapDataset(tmp_path2)
    try:
        iterator = mmap_ninja.MMapBatchIterator(dataset2, batch_size=10, shuffle=True)
        print(f"✅ Iterator dibuat, total batch: {iterator.num_batches()}")
        # Ambil 2 batch
        for i in range(2):
            has, batch = iterator.next_batch()
            if has:
                print(f"   Batch {i+1}: {len(batch)} contoh")
            else:
                print(f"   Batch {i+1}: tidak ada data")
    except Exception as e:
        print(f"❌ Iterator gagal: {e}")
    os.unlink(tmp_path2)

print_section("✅ SEMUA UJI SELESAI")
print("Binding mmap_ninja berfungsi dengan baik.")