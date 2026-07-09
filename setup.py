from setuptools import setup, Extension
import pybind11

ext = Extension(
    'minigpt',
    ['minigpt.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3']
)

setup(name='minigpt', version='0.1', ext_modules=[ext])