from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Jika dijalankan tanpa argumen, otomatis build in-place
if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

# Daftar semua file .cpp yang diperlukan (kecuali main.cpp, train.cpp, dan Value.cpp duplikat)
sources = [
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
    "minigpt/value.cpp",   # huruf kecil, bukan Value.cpp
]

ext_modules = [
    Pybind11Extension(
        "minigpt",
        sources=sources,
        include_dirs=["minigpt"],
        cxx_std=17,
        # extra_compile_args=["-O3", "-march=native"],  # aktifkan jika perlu
    ),
]

setup(
    name="minigpt",
    version="0.1.0",
    description="MiniGPT Python Extension",
    author="MiniGPT",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)