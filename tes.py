#!/usr/bin/env python3
"""
tes.py - Memeriksa hasil build binding untuk mmap_ninja (di dalam modul minigpt)
"""
import os
import sys
import importlib
import tempfile
import random

def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

# ============================================================
# 1. Cek modul minigpt
# ============================================================
print_section("1. IMPOR MODUL minigpt & mmap_ninja")
try:
    import minigpt
    print("✅ Modul 'minigpt' berhasil diimpor.")
except ImportError as e:
    print(f"❌ Gagal mengimpor minigpt: {e}")
    print("   Jalankan ulang: python3 setup.py build_ext --inplace")
    sys.exit(1)

# ============================================================
# 2. Cek apakah mmap_ninja ada di dalam minigpt
# ============================================================
print_section("2. ATRIBUT minigpt (Cari MMapDataset & build)")
attrs = [attr for attr in dir(minigpt) if not attr.startswith('_')]
print(f"Ditemukan {len(attrs)} atribut publik:")

# Cari nama fungsi/kelas yang terkait dengan mmap
mmap_attrs = [a for a in attrs if 'mmap' in a.lower() or 'MMap' in a or 'build' in a]
if mmap_attrs:
    print(f"\n🔍 Atribut yang berhubungan dengan mmap/build:")
    for attr in mmap_attrs:
        obj = getattr(minigpt, attr)
        print(f"  - {attr:<20} : {type(obj).__name__}")
else:
    print("\n⚠️  Tidak ada atribut 'MMap' atau 'build' yang terdeteksi.")
    print("   Daftar lengkap atribut yang tersedia:")
    for attr in attrs[:20]:  # tampilkan 20 pertama
        print(f"  - {attr}")

# ============================================================
# 3. Coba import langsung (sesuai dengan training.py)
# ============================================================
print_section("3. IMPOR LANGSUNG (seperti di training.py)")
try:
    # Ini sesuai dengan baris di training.py Anda:
    # from minigpt import MMapDataset, build_mmap_dataset
    from minigpt import MMapDataset, build_mmap_dataset
    print("✅ Berhasil import MMapDataset dan build_mmap_dataset")
    
    # Cek tipe
    print(f"   MMapDataset       : {type(MMapDataset)}")
    print(f"   build_mmap_dataset: {type(build_mmap_dataset)}")
    
except ImportError as e:
    print(f"❌ Gagal import spesifik: {e}")
    print("   Kemungkinan nama fungsi di bindings.cpp berbeda.")
    print("   Coba cek daftar atribut di bagian 2 di atas.")
    
    # Coba fallback: cari fungsi yang namanya mirip
    print("\n🔄 Mencoba fallback...")
    if hasattr(minigpt, 'build'):
        print("   - Ditemukan 'minigpt.build' (coba gunakan ini)")
        build_func = minigpt.build
        print(f"     build: {build_func}")
    if hasattr(minigpt, 'MMapDataset'):
        print("   - Ditemukan 'minigpt.MMapDataset'")
    
    sys.exit(1)

# ============================================================
# 4. UJI FUNGSIONALITAS DASAR
# ============================================================
print_section("4. UJI FUNGSIONALITAS DASAR (tulis & baca)")
SEQ_LEN = 8
NUM_EXAMPLES = 50
dummy_data = [[random.randint(0, 100) for _ in range(SEQ_LEN)] for _ in range(NUM_EXAMPLES)]
print(f"Data dummy: {NUM_EXAMPLES} contoh, seq_len={SEQ_LEN}")

with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
    tmp_path = tmp.name

print(f"Menulis ke: {tmp_path}")
try:
    build_mmap_dataset(tmp_path, dummy_data, SEQ_LEN)
    print("✅ build_mmap_dataset() berhasil.")
except Exception as e:
    print(f"❌ build_mmap_dataset() gagal: {e}")
    sys.exit(1)

try:
    dataset = MMapDataset(tmp_path)
    print(f"✅ MMapDataset berhasil dibuat.")
    print(f"   Jumlah contoh: {dataset.size()}")
    
    # Ambil sample
    first = dataset.get_example(0)
    print(f"   Contoh pertama (5 angka): {first[:5]}")
    
    batch = dataset.get_batch([0, 10, 49])
    print(f"   Batch 3 contoh: {len(batch)} baris, panjang {len(batch[0])}")
    
except Exception as e:
    print(f"❌ Gagal membaca: {e}")
    sys.exit(1)

os.unlink(tmp_path)
print("🗑️  File sementara dihapus.")

# ============================================================
# 5. UJI BATCH ITERATOR (jika ada)
# ============================================================
if hasattr(minigpt, 'MMapBatchIterator'):
    print_section("5. UJI MMapBatchIterator")
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path2 = tmp.name
    build_mmap_dataset(tmp_path2, dummy_data, SEQ_LEN)
    dataset2 = MMapDataset(tmp_path2)
    try:
        iterator = minigpt.MMapBatchIterator(dataset2, batch_size=10, shuffle=True)
        print(f"✅ Iterator dibuat, total batch: {iterator.num_batches()}")
        has, batch = iterator.next_batch()
        if has:
            print(f"   Batch pertama: {len(batch)} contoh")
    except Exception as e:
        print(f"❌ Iterator gagal: {e}")
    os.unlink(tmp_path2)

print_section("✅ SEMUA UJI SELESAI")
print("Binding di dalam modul minigpt berfungsi dengan baik.")