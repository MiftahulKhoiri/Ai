from setuptools import setup, Extension
import pybind11

ext = Extension(
    'minigpt',
    sources=[
        'minigpt/value.cpp',
        'minigpt/utils.cpp',
        'minigpt/tokenizer.cpp',
        'minigpt/layers.cpp',
        'minigpt/model.cpp',
        'minigpt/optim.cpp',
        'minigpt/generation.cpp',
        'minigpt/bindings.cpp'
    ],
    include_dirs=[pybind11.get_include(), 'minigpt'],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3']
)

setup(name='minigpt', version='0.1', ext_modules=[ext])