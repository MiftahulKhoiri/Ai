#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "value.h"
#include "tokenizer.h"
#include "layers.h"
#include "model.h"
#include "optim.h"
#include "generation.h"

namespace py = pybind11;

PYBIND11_MODULE(minigpt, m) {
    m.doc() = "MiniGPT C++ implementation with autograd and BPE tokenizer";

    // ============================================================
    // Value - Autograd Engine
    // ============================================================
    py::class_<Value, ValuePtr>(m, "Value")
        .def(py::init<double>())
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def("backward", &Value::backward)
        .def("__repr__", &Value::repr)
        // Arithmetic operators
        .def("__add__", [](ValuePtr a, ValuePtr b) { return a + b; })
        .def("__add__", [](ValuePtr a, double b) { return a + b; })
        .def("__radd__", [](ValuePtr a, double b) { return b + a; })
        .def("__mul__", [](ValuePtr a, ValuePtr b) { return a * b; })
        .def("__mul__", [](ValuePtr a, double b) { return a * b; })
        .def("__rmul__", [](ValuePtr a, double b) { return b * a; })
        .def("__sub__", [](ValuePtr a, ValuePtr b) { return a - b; })
        .def("__sub__", [](ValuePtr a, double b) { return a - b; })
        .def("__rsub__", [](ValuePtr a, double b) { return b - a; })
        .def("__truediv__", [](ValuePtr a, ValuePtr b) { return a / b; })
        .def("__truediv__", [](ValuePtr a, double b) { return a / b; })
        .def("__rtruediv__", [](ValuePtr a, double b) { return b / a; })
        .def("__pow__", [](ValuePtr a, double exp) { return pow(a, exp); })
        // Activation functions
        .def("exp", [](ValuePtr a) { return exp(a); })
        .def("log", [](ValuePtr a) { return log(a); })
        .def("sqrt", [](ValuePtr a) { return sqrt(a); })
        .def("tanh", [](ValuePtr a) { return tanh(a); })
        .def("relu", [](ValuePtr a) { return relu(a); })
        .def("gelu", [](ValuePtr a) { return gelu(a); });

    // ============================================================
    // Tokenizer
    // ============================================================
    py::class_<ByteLevelBPETokenizer>(m, "ByteLevelBPETokenizer")
        .def(py::init<>())
        .def("train", &ByteLevelBPETokenizer::train,
             py::arg("corpus"), py::arg("vocab_size") = 400)
        .def("encode", &ByteLevelBPETokenizer::encode,
             py::arg("text"), py::arg("add_bos") = false, py::arg("add_eos") = false)
        .def("decode", &ByteLevelBPETokenizer::decode)
        .def("save", &ByteLevelBPETokenizer::save)
        .def("load", &ByteLevelBPETokenizer::load)
        // ===== PERUBAHAN: Gunakan getter/setter =====
        .def_property("vocab",
            &ByteLevelBPETokenizer::get_vocab,
            &ByteLevelBPETokenizer::set_vocab)
        .def_property("inv_vocab",
            &ByteLevelBPETokenizer::get_inv_vocab,
            &ByteLevelBPETokenizer::set_inv_vocab)
        .def_property("merge_order",
            &ByteLevelBPETokenizer::get_merge_order,
            &ByteLevelBPETokenizer::set_merge_order);

    // ============================================================
    // Layers
    // ============================================================

    // Dropout
    py::class_<Dropout>(m, "Dropout")
        .def(py::init<double>(), py::arg("p") = 0.1)
        .def("forward", &Dropout::forward)
        .def_readwrite("training", &Dropout::training);

    // Linear
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int, bool>(),
             py::arg("n_in"), py::arg("n_out"), py::arg("bias") = true)
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters);

    // Embedding
    py::class_<Embedding>(m, "Embedding")
        .def(py::init<int, int, double>(),
             py::arg("vocab_size"), py::arg("d_model"), py::arg("scale") = 0.02)
        .def("forward", &Embedding::forward)
        .def("parameters", &Embedding::parameters);

    // PositionalEmbedding
    py::class_<PositionalEmbedding>(m, "PositionalEmbedding")
        .def(py::init<int, int, double>(),
             py::arg("max_len"), py::arg("d_model"), py::arg("scale") = 0.02)
        .def("forward", &PositionalEmbedding::forward)
        .def("parameters", &PositionalEmbedding::parameters);

    // LayerNorm
    py::class_<LayerNorm>(m, "LayerNorm")
        .def(py::init<int, double>(),
             py::arg("dim"), py::arg("eps") = 1e-5)
        .def("forward", &LayerNorm::forward)
        .def("parameters", &LayerNorm::parameters);

    // MultiHeadSelfAttention
    py::class_<MultiHeadSelfAttention>(m, "MultiHeadSelfAttention")
        .def(py::init<int, int, double>(),
             py::arg("d_model"), py::arg("n_heads"), py::arg("dropout") = 0.1)
        .def("forward", &MultiHeadSelfAttention::forward)
        .def("forward_incremental", &MultiHeadSelfAttention::forward_incremental)
        .def("parameters", &MultiHeadSelfAttention::parameters)
        .def("set_training", &MultiHeadSelfAttention::set_training);

    // FeedForward
    py::class_<FeedForward>(m, "FeedForward")
        .def(py::init<int, int, double>(),
             py::arg("d_model"), py::arg("d_ff"), py::arg("dropout") = 0.1)
        .def("forward", &FeedForward::forward)
        .def("parameters", &FeedForward::parameters)
        .def("set_training", &FeedForward::set_training);

    // TransformerBlock
    py::class_<TransformerBlock>(m, "TransformerBlock")
        .def(py::init<int, int, int, double>(),
             py::arg("d_model"), py::arg("n_heads"), py::arg("d_ff"), py::arg("dropout") = 0.1)
        .def("forward", &TransformerBlock::forward)
        .def("forward_incremental", &TransformerBlock::forward_incremental)
        .def("parameters", &TransformerBlock::parameters)
        .def("set_training", &TransformerBlock::set_training);

    // ============================================================
    // MiniGPT Model
    // ============================================================
    py::class_<MiniGPT>(m, "MiniGPT")
        .def(py::init<int, int, int, int, int, int, double>(),
             py::arg("vocab_size"), 
             py::arg("d_model") = 16, 
             py::arg("n_heads") = 2,
             py::arg("n_layers") = 2, 
             py::arg("d_ff") = 32, 
             py::arg("max_len") = 64,
             py::arg("dropout") = 0.1)
        .def("forward", &MiniGPT::forward,
             py::arg("token_ids"), 
             py::arg("pad_mask") = std::vector<int>())
        .def("init_cache", &MiniGPT::init_cache)
        .def("forward_incremental", &MiniGPT::forward_incremental)
        .def("parameters", &MiniGPT::parameters)
        .def("set_training", &MiniGPT::set_training)
        .def_readwrite("d_model", &MiniGPT::d_model)
        .def_readwrite("max_len", &MiniGPT::max_len)
        .def_readwrite("vocab_size", &MiniGPT::vocab_size);

    // ============================================================
    // Optimizer & Loss
    // ============================================================
    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<ValuePtr>, double, double, double, double, double>(),
             py::arg("params"), 
             py::arg("lr") = 0.01, 
             py::arg("betas1") = 0.9,
             py::arg("betas2") = 0.999, 
             py::arg("eps") = 1e-8, 
             py::arg("weight_decay") = 0.01)
        .def("step", &AdamW::step)
        .def("zero_grad", &AdamW::zero_grad)
        .def_readwrite("lr", &AdamW::lr)
        // ===== PERUBAHAN: Gunakan getter/setter =====
        .def_property("params",
            &AdamW::get_params,
            &AdamW::set_params)
        .def_property("m",
            &AdamW::get_m,
            &AdamW::set_m)
        .def_property("v",
            &AdamW::get_v,
            &AdamW::set_v)
        .def_property("t",
            &AdamW::get_t,
            &AdamW::set_t);

    // Scheduler
    py::class_<WarmupCosineScheduler>(m, "WarmupCosineScheduler")
        .def(py::init<AdamW*, int, int, double, double>(),
             py::arg("optimizer"), 
             py::arg("warmup_steps"), 
             py::arg("total_steps"),
             py::arg("base_lr"), 
             py::arg("min_lr") = 1e-5)
        .def("step", &WarmupCosineScheduler::step)
        // ===== PERUBAHAN: Gunakan getter/setter =====
        .def_property("step_num",
            &WarmupCosineScheduler::get_step_num,
            &WarmupCosineScheduler::set_step_num);

    // Loss and utility functions
    m.def("cross_entropy_loss", &cross_entropy_loss,
          py::arg("logits_seq"), 
          py::arg("target_ids"), 
          py::arg("pad_mask"));

    m.def("clip_grad_norm", &clip_grad_norm,
          py::arg("params"), 
          py::arg("max_norm"));

    m.def("sample_from_logits", &sample_from_logits,
          py::arg("logits"), 
          py::arg("temperature"), 
          py::arg("top_k"), 
          py::arg("top_p"));

    // ============================================================
    // Generation
    // ============================================================
    m.def("generate", &generate,
          py::arg("model"), 
          py::arg("tokenizer"), 
          py::arg("prompt"),
          py::arg("max_new_tokens") = 25, 
          py::arg("temperature") = 0.9,
          py::arg("top_k") = 0, 
          py::arg("top_p") = 0.9);
}