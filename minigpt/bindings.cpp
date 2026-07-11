// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "generation.h"

namespace py = pybind11;

PYBIND11_MODULE(minigpt, m) {
    m.doc() = "MiniGPT - Lightweight GPT implementation with autograd";

    // ===== BINDING UNTUK VALUE =====
    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init<double>())
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def("backward", &Value::backward)
        .def("__repr__", &Value::repr)
        .def("__add__", [](const Value::Ptr& a, const Value::Ptr& b) { return a + b; })
        .def("__sub__", [](const Value::Ptr& a, const Value::Ptr& b) { return a - b; })
        .def("__mul__", [](const Value::Ptr& a, const Value::Ptr& b) { return a * b; })
        .def("__truediv__", [](const Value::Ptr& a, const Value::Ptr& b) { return a / b; });

    // ===== BINDING UNTUK TOKENIZER =====
    py::class_<ByteLevelBPETokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("train", &ByteLevelBPETokenizer::train)
        .def("encode", &ByteLevelBPETokenizer::encode, 
             py::arg("text"), py::arg("add_bos") = false, py::arg("add_eos") = false)
        .def("decode", &ByteLevelBPETokenizer::decode)
        .def("save", &ByteLevelBPETokenizer::save)
        .def("load", &ByteLevelBPETokenizer::load)
        .def("vocab_size", &ByteLevelBPETokenizer::vocab_size)
        .def("get_vocab", &ByteLevelBPETokenizer::get_vocab)
        .def("get_inv_vocab", &ByteLevelBPETokenizer::get_inv_vocab)
        .def("get_merge_order", &ByteLevelBPETokenizer::get_merge_order);

    // ===== BINDING UNTUK MINIGPT MODEL =====
    py::class_<MiniGPT>(m, "MiniGPT")
        .def(py::init<int, int, int, int, int, int, double>(),
             py::arg("vocab_size"),
             py::arg("d_model") = 16,
             py::arg("n_heads") = 2,
             py::arg("n_layers") = 2,
             py::arg("d_ff") = 32,
             py::arg("max_len") = 64,
             py::arg("dropout") = 0.1)
        .def("forward", &MiniGPT::forward)
        .def("forward_incremental", &MiniGPT::forward_incremental)
        .def("init_cache", &MiniGPT::init_cache)
        .def("parameters", &MiniGPT::parameters)
        .def("set_training", &MiniGPT::set_training)
        .def_readonly("d_model", &MiniGPT::d_model)
        .def_readonly("max_len", &MiniGPT::max_len)
        .def_readonly("vocab_size", &MiniGPT::vocab_size);

    // ===== BINDING UNTUK OPTIMIZER =====
    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<Value::Ptr>, double, double, double, double, double, bool>(),
             py::arg("params"),
             py::arg("lr") = 1e-3,
             py::arg("betas1") = 0.9,
             py::arg("betas2") = 0.999,
             py::arg("eps") = 1e-8,
             py::arg("weight_decay") = 0.01,
             py::arg("decoupled_wd") = true)
        .def("step", &AdamW::step)
        .def("zero_grad", &AdamW::zero_grad)
        .def_readwrite("lr", &AdamW::lr);

    py::class_<WarmupCosineScheduler>(m, "WarmupCosineScheduler")
        .def(py::init<AdamW*, int, int, double, double>(),
             py::arg("opt"),
             py::arg("warmup_steps"),
             py::arg("total_steps"),
             py::arg("base_lr") = 1e-3,
             py::arg("min_lr") = 1e-5)
        .def("step", &WarmupCosineScheduler::step);

    // ===== BINDING UNTUK LOSS FUNCTIONS =====
    m.def("cross_entropy_loss", &cross_entropy_loss,
          py::arg("logits_seq"), py::arg("target_ids"), py::arg("pad_mask") = std::vector<int>{});

    m.def("clip_grad_norm", &clip_grad_norm,
          py::arg("params"), py::arg("max_norm"));

    // ===== BINDING UNTUK GENERATION FUNCTIONS =====
    // Argmax function
    m.def("argmax", &argmax, py::arg("logits"),
          "Get the index of the maximum value in logits");
    
    // Sample from logits with top-k and temperature
    m.def("sample_top_k", &sample_top_k, 
          py::arg("logits"), py::arg("k") = 0, py::arg("temperature") = 1.0f,
          "Sample from logits with top-k filtering and temperature");

    // Generate function - PERBAIKI JUMLAH ARGUMEN!
    m.def("generate", &generate,
          py::arg("model"),
          py::arg("tokenizer"),
          py::arg("prompt"),
          py::arg("max_tokens") = 50,
          py::arg("add_bos") = false,
          py::arg("add_eos") = false,
          "Generate text from a prompt");
}