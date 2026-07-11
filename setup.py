from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

# Jika dijalankan tanpa argumen, otomatis build in-place
if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

# Daftar semua file .cpp yang diperlukan
sources = [
    # Core components
    "minigpt/autograd.cpp",
    "minigpt/backward.cpp",
    "minigpt/bindings.cpp",
    "minigpt/generation.cpp",
    "minigpt/graph.cpp",
    "minigpt/layers.cpp",
    "minigpt/matrix.cpp",
    "minigpt/model.cpp",
    "minigpt/ops.cpp",
    "minigpt/optim.cpp",
    "minigpt/random.cpp",
    "minigpt/simd.cpp",
    "minigpt/tensor.cpp",
    "minigpt/tokenizer.cpp",
    "minigpt/utils.cpp",
    "minigpt/value.cpp",
    
    # New features (Tahap 1-10)
    "minigpt/config.cpp",           # Configuration system
    "minigpt/dataloader.cpp",       # DataLoader
    "minigpt/sampling.cpp",         # Advanced sampling (Top-K, Top-P)
    "minigpt/metrics.cpp",          # Metrics & evaluation
    "minigpt/checkpoint.cpp",       # Checkpoint manager
    "minigpt/generation_advanced.cpp", # Beam search, repetition penalty
    "minigpt/visualization.cpp",    # Visualization tools
    "minigpt/test.cpp",             # Unit testing framework
    "minigpt/server.cpp",           # API server
    "minigpt/quantization.cpp",     # Quantization & pruning
]

# Cek file yang ada vs yang dibutuhkan
missing_files = []
for source in sources:
    if not os.path.exists(source):
        missing_files.append(source)

if missing_files:
    print("⚠️  Warning: Beberapa file tidak ditemukan:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n📝  File-file baru perlu dibuat. Build akan dilanjutkan dengan file yang ada.")
    
    # Filter hanya file yang ada
    sources = [f for f in sources if os.path.exists(f)]
    if not sources:
        print("❌ Tidak ada file sumber ditemukan!")
        sys.exit(1)

ext_modules = [
    Pybind11Extension(
        "minigpt",
        sources=sources,
        include_dirs=["minigpt"],
        cxx_std=17,
        # Tambahkan flag kompilasi untuk optimasi
        extra_compile_args=[
            "-O3",
            "-Wall",
            "-Wextra",
            # "-march=native",  # Aktifkan jika di x86
        ],
        # Untuk linking dengan library tambahan
        # libraries=["pthread"],
    ),
]

setup(
    name="minigpt",
    version="0.2.0",
    description="MiniGPT - Complete C++ GPT Implementation with Python Bindings",
    author="MiniGPT",
    author_email="minigpt@example.com",
    url="https://github.com/yourusername/minigpt",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)