from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Jika dijalankan tanpa argumen:
# python setup.py
# otomatis menjadi:
# python setup.py build_ext --inplace
if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

ext_modules = [
    Pybind11Extension(
        "minigpt",
        [
            "minigpt/value.cpp",
            "minigpt/utils.cpp",
            "minigpt/tokenizer.cpp",
            "minigpt/layers.cpp",
            "minigpt/model.cpp",
            "minigpt/optim.cpp",
            "minigpt/generation.cpp",
            "minigpt/bindings.cpp",
        ],
        include_dirs=[
            "minigpt",
        ],
        cxx_std=17,
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