// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "model.h"
#include "tokenizer.h"
#include "optim.h"
#include "generation.h"
#include "config.h"
#include "dataloader.h"
#include "sampling.h"
#include "metrics.h"
#include "checkpoint.h"
#include "generation_advanced.h"
#include "visualization.h"
#include "test.h"
#include "quantization.h"

namespace py = pybind11;

PYBIND11_MODULE(minigpt, m) {
    m.doc() = "MiniGPT - Complete C++ GPT Implementation with Python Bindings";

    // ============================================================
    // VALUE
    // ============================================================
    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init<double>(), py::arg("data"))
        .def_readwrite("data", &Value::data)
        .def_readwrite("grad", &Value::grad)
        .def("backward", &Value::backward)
        .def("__repr__", &Value::repr)
        .def("__add__", [](const Value::Ptr& a, const Value::Ptr& b) { return a + b; })
        .def("__sub__", [](const Value::Ptr& a, const Value::Ptr& b) { return a - b; })
        .def("__mul__", [](const Value::Ptr& a, const Value::Ptr& b) { return a * b; })
        .def("__truediv__", [](const Value::Ptr& a, const Value::Ptr& b) { return a / b; });

    // ============================================================
    // TOKENIZER
    // ============================================================
    py::class_<ByteLevelBPETokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("train", &ByteLevelBPETokenizer::train, 
             py::arg("corpus"), py::arg("vocab_size"))
        .def("encode", &ByteLevelBPETokenizer::encode, 
             py::arg("text"), py::arg("add_bos") = false, py::arg("add_eos") = false)
        .def("decode", &ByteLevelBPETokenizer::decode, py::arg("ids"))
        .def("save", &ByteLevelBPETokenizer::save, py::arg("path"))
        .def("load", &ByteLevelBPETokenizer::load, py::arg("path"))
        .def("vocab_size", &ByteLevelBPETokenizer::vocab_size)
        .def("get_vocab", &ByteLevelBPETokenizer::get_vocab)
        .def("get_inv_vocab", &ByteLevelBPETokenizer::get_inv_vocab)
        .def("get_merge_order", &ByteLevelBPETokenizer::get_merge_order)
        .def("get_eos_token_id", &ByteLevelBPETokenizer::get_eos_token_id)
        .def("get_bos_token_id", &ByteLevelBPETokenizer::get_bos_token_id)
        .def("get_pad_token_id", &ByteLevelBPETokenizer::get_pad_token_id)
        .def("get_unk_token_id", &ByteLevelBPETokenizer::get_unk_token_id)
        .def("set_vocab", &ByteLevelBPETokenizer::set_vocab, py::arg("vocab"))
        .def("set_inv_vocab", &ByteLevelBPETokenizer::set_inv_vocab, py::arg("inv_vocab"))
        .def("set_merge_order", &ByteLevelBPETokenizer::set_merge_order, py::arg("merge_order"));

    // ============================================================
    // MODEL CONFIG
    // ============================================================
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("d_model", &ModelConfig::d_model)
        .def_readwrite("n_heads", &ModelConfig::n_heads)
        .def_readwrite("n_layers", &ModelConfig::n_layers)
        .def_readwrite("d_ff", &ModelConfig::d_ff)
        .def_readwrite("max_len", &ModelConfig::max_len)
        .def_readwrite("dropout", &ModelConfig::dropout)
        .def_readwrite("layer_norm_eps", &ModelConfig::layer_norm_eps)
        .def_readwrite("learning_rate", &ModelConfig::learning_rate)
        .def_readwrite("weight_decay", &ModelConfig::weight_decay)
        .def_readwrite("beta1", &ModelConfig::beta1)
        .def_readwrite("beta2", &ModelConfig::beta2)
        .def_readwrite("eps", &ModelConfig::eps)
        .def_readwrite("warmup_steps", &ModelConfig::warmup_steps)
        .def_readwrite("total_steps", &ModelConfig::total_steps)
        .def_readwrite("batch_size", &ModelConfig::batch_size)
        .def_readwrite("epochs", &ModelConfig::epochs)
        .def_readwrite("temperature", &ModelConfig::temperature)
        .def_readwrite("top_k", &ModelConfig::top_k)
        .def_readwrite("top_p", &ModelConfig::top_p)
        .def_readwrite("use_cache", &ModelConfig::use_cache)
        .def_readwrite("activation", &ModelConfig::activation)
        .def_readwrite("use_bias", &ModelConfig::use_bias)
        .def_readwrite("init_std", &ModelConfig::init_std)
        .def_readwrite("share_embeddings", &ModelConfig::share_embeddings)
        .def_readwrite("device", &ModelConfig::device)
        .def_static("from_json", &ModelConfig::from_json, py::arg("path"))
        .def("to_json", &ModelConfig::to_json, py::arg("path"));

    // ============================================================
    // MINIGPT MODEL
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
             py::arg("token_ids"), py::arg("pad_mask") = std::vector<int>{})
        .def("forward_incremental", &MiniGPT::forward_incremental, 
             py::arg("token_id"), py::arg("pos"))
        .def("init_cache", &MiniGPT::init_cache)
        .def("parameters", &MiniGPT::parameters)
        .def("set_training", &MiniGPT::set_training, py::arg("mode"))
        .def_readonly("d_model", &MiniGPT::d_model)
        .def_readonly("max_len", &MiniGPT::max_len)
        .def_readonly("vocab_size", &MiniGPT::vocab_size)
        .def("__repr__", [](const MiniGPT& model) {
            return "<MiniGPT d_model=" + std::to_string(model.d_model) + 
                   " layers=" + std::to_string(model.blocks.size()) +
                   " vocab=" + std::to_string(model.vocab_size) + ">";
        });

    // ============================================================
    // OPTIMIZER
    // ============================================================
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
        .def_readwrite("lr", &AdamW::lr)
        .def("get_params", &AdamW::get_params)
        .def("get_m", &AdamW::get_m)
        .def("get_v", &AdamW::get_v)
        .def("get_t", &AdamW::get_t)
        .def("set_params", &AdamW::set_params, py::arg("params"))
        .def("set_m", &AdamW::set_m, py::arg("m"))
        .def("set_v", &AdamW::set_v, py::arg("v"))
        .def("set_t", &AdamW::set_t, py::arg("t"));

    py::class_<WarmupCosineScheduler>(m, "WarmupCosineScheduler")
        .def(py::init<AdamW*, int, int, double, double>(),
             py::arg("opt"),
             py::arg("warmup_steps"),
             py::arg("total_steps"),
             py::arg("base_lr") = 1e-3,
             py::arg("min_lr") = 1e-5)
        .def("step", &WarmupCosineScheduler::step)
        .def("get_step_num", &WarmupCosineScheduler::get_step_num)
        .def("set_step_num", &WarmupCosineScheduler::set_step_num, py::arg("step_num"));

    // ============================================================
    // LOSS FUNCTIONS
    // ============================================================
    m.def("cross_entropy_loss", &cross_entropy_loss,
          py::arg("logits_seq"), py::arg("target_ids"), py::arg("pad_mask") = std::vector<int>{});

    m.def("log_softmax", &log_softmax, py::arg("x"));

    m.def("clip_grad_norm", &clip_grad_norm,
          py::arg("params"), py::arg("max_norm"));

    // ============================================================
    // GENERATION
    // ============================================================
    m.def("generate", &generate,
          py::arg("model"),
          py::arg("tokenizer"),
          py::arg("prompt"),
          py::arg("max_tokens") = 50,
          py::arg("add_bos") = false,
          py::arg("add_eos") = false);

    m.def("argmax", &argmax, py::arg("logits"));
    m.def("sample_top_k", &sample_top_k, 
          py::arg("logits"), py::arg("k") = 0, py::arg("temperature") = 1.0f);

    // ============================================================
    // DATALOADER
    // ============================================================
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<ByteLevelBPETokenizer&, int, int>(),
             py::arg("tokenizer"),
             py::arg("batch_size") = 8,
             py::arg("seq_len") = 128)
        .def("load_text", &DataLoader::load_text,
             py::arg("text"), py::arg("add_bos") = true, py::arg("add_eos") = true)
        .def("load_texts", &DataLoader::load_texts,
             py::arg("texts"), py::arg("add_bos") = true, py::arg("add_eos") = true)
        .def("next_batch", &DataLoader::next_batch)
        .def("shuffle", &DataLoader::shuffle)
        .def("num_batches", &DataLoader::num_batches)
        .def("reset", &DataLoader::reset)
        .def("size", &DataLoader::size);

    // ============================================================
    // SAMPLING
    // ============================================================
    m.def("greedy_sample", &sampling::greedy_sample, py::arg("logits"));
    m.def("top_k_sample", &sampling::top_k_sample, 
          py::arg("logits"), py::arg("k"), py::arg("temperature") = 1.0f);
    m.def("top_p_sample", &sampling::top_p_sample,
          py::arg("logits"), py::arg("p"), py::arg("temperature") = 1.0f);
    m.def("batch_sample", &sampling::batch_sample,
          py::arg("logits_batch"), py::arg("method") = "top_p",
          py::arg("temperature") = 0.8f, py::arg("top_k") = 40, py::arg("top_p") = 0.9f);

    // ============================================================
    // METRICS
    // ============================================================
    m.def("perplexity", &metrics::perplexity, py::arg("loss"));
    m.def("accuracy", &metrics::accuracy, py::arg("predictions"), py::arg("targets"));
    m.def("bleu_score", &metrics::bleu_score, py::arg("reference"), py::arg("candidate"));

    py::class_<metrics::TrainingLogger>(m, "TrainingLogger")
        .def(py::init<const std::string&>(), py::arg("name") = "training")
        .def("log_epoch", &metrics::TrainingLogger::log_epoch,
             py::arg("epoch"), py::arg("loss"), py::arg("accuracy") = 0.0, py::arg("lr") = 0.0)
        .def("log_metric", &metrics::TrainingLogger::log_metric, py::arg("key"), py::arg("value"))
        .def("save_logs", &metrics::TrainingLogger::save_logs, py::arg("path") = "logs.csv")
        .def("print_summary", &metrics::TrainingLogger::print_summary)
        .def("get_losses", &metrics::TrainingLogger::get_losses)
        .def("get_accuracies", &metrics::TrainingLogger::get_accuracies)
        .def("get_lrs", &metrics::TrainingLogger::get_lrs);

    py::class_<metrics::ProgressBar>(m, "ProgressBar")
        .def(py::init<int, const std::string&>(), py::arg("total"), py::arg("desc") = "Progress")
        .def("update", &metrics::ProgressBar::update, py::arg("current"))
        .def("increment", &metrics::ProgressBar::increment)
        .def("finish", &metrics::ProgressBar::finish);

    // ============================================================
    // CHECKPOINT MANAGER
    // ============================================================
    py::class_<CheckpointManager>(m, "CheckpointManager")
        .def(py::init<const std::string&>(), py::arg("save_dir") = "checkpoints")
        .def("save", &CheckpointManager::save,
             py::arg("model"), py::arg("optimizer"), py::arg("epoch"), py::arg("loss"), py::arg("name") = "")
        .def("load", &CheckpointManager::load,
             py::arg("model"), py::arg("optimizer"), py::arg("epoch"), py::arg("loss"), py::arg("name") = "")
        .def("set_auto_save", &CheckpointManager::set_auto_save, py::arg("interval"))
        .def("set_max_keep", &CheckpointManager::set_max_keep, py::arg("max_keep"))
        .def("get_latest_checkpoint", &CheckpointManager::get_latest_checkpoint)
        .def("list_checkpoints", &CheckpointManager::list_checkpoints)
        .def("clean_old_checkpoints", &CheckpointManager::clean_old_checkpoints);

    // ============================================================
    // ADVANCED GENERATION
    // ============================================================
    py::class_<advanced_generation::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_length", &advanced_generation::GenerationConfig::max_length)
        .def_readwrite("min_length", &advanced_generation::GenerationConfig::min_length)
        .def_readwrite("num_beams", &advanced_generation::GenerationConfig::num_beams)
        .def_readwrite("temperature", &advanced_generation::GenerationConfig::temperature)
        .def_readwrite("top_p", &advanced_generation::GenerationConfig::top_p)
        .def_readwrite("top_k", &advanced_generation::GenerationConfig::top_k)
        .def_readwrite("repetition_penalty", &advanced_generation::GenerationConfig::repetition_penalty)
        .def_readwrite("length_penalty", &advanced_generation::GenerationConfig::length_penalty)
        .def_readwrite("num_return_sequences", &advanced_generation::GenerationConfig::num_return_sequences)
        .def_readwrite("use_cache", &advanced_generation::GenerationConfig::use_cache)
        .def_readwrite("early_stopping", &advanced_generation::GenerationConfig::early_stopping)
        .def_readwrite("no_repeat_ngram_size", &advanced_generation::GenerationConfig::no_repeat_ngram_size);

    py::class_<advanced_generation::AdvancedGenerator>(m, "AdvancedGenerator")
        .def(py::init<MiniGPT&, ByteLevelBPETokenizer&>())
        .def("set_config", &advanced_generation::AdvancedGenerator::set_config)
        .def("get_config", &advanced_generation::AdvancedGenerator::get_config)
        .def("generate", &advanced_generation::AdvancedGenerator::generate,
             py::arg("prompt"), py::arg("add_bos") = false, py::arg("add_eos") = false);

    // ============================================================
    // VISUALIZATION
    // ============================================================
    m.def("graph_to_dot", &visualization::graph_to_dot, 
          py::arg("node"), py::arg("name") = "graph");
    m.def("plot_attention", &visualization::plot_attention,
          py::arg("attention"), py::arg("filename") = "attention.csv");
    m.def("save_embeddings", &visualization::save_embeddings,
          py::arg("embeddings"), py::arg("labels"), py::arg("filename") = "embeddings.csv");

    py::class_<visualization::TrainingMonitor>(m, "TrainingMonitor")
        .def(py::init<>())
        .def("add_metric", &visualization::TrainingMonitor::add_metric)
        .def("new_epoch", &visualization::TrainingMonitor::new_epoch)
        .def("save_to_csv", &visualization::TrainingMonitor::save_to_csv)
        .def("print_summary", &visualization::TrainingMonitor::print_summary);

    // ============================================================
    // QUANTIZATION
    // ============================================================
    m.def("quantize_float_to_int8", &quantization::quantize_float_to_int8,
          py::arg("data"), py::arg("quantized"), py::arg("scale"), py::arg("zero_point"));
    m.def("dequantize_int8_to_float", &quantization::dequantize_int8_to_float,
          py::arg("quantized"), py::arg("data"), py::arg("scale"), py::arg("zero_point"));
    m.def("prune_weights", &quantization::prune_weights,
          py::arg("model"), py::arg("threshold") = 0.01);
    m.def("compress_model", &quantization::compress_model,
          py::arg("model"), py::arg("compression_ratio") = 0.5);

    py::class_<quantization::DistillationConfig>(m, "DistillationConfig")
        .def(py::init<>())
        .def_readwrite("temperature", &quantization::DistillationConfig::temperature)
        .def_readwrite("alpha", &quantization::DistillationConfig::alpha)
        .def_readwrite("beta", &quantization::DistillationConfig::beta);

    m.def("distill", &quantization::distill,
          py::arg("teacher"), py::arg("student"), py::arg("dataloader"), 
          py::arg("optimizer"), py::arg("config") = quantization::DistillationConfig());

    // ============================================================
    // TEST RUNNER
    // ============================================================
    m.def("run_all_tests", &run_all_tests, "Run all unit tests");

    // ============================================================
    // VERSION INFO
    // ============================================================
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "MiniGPT Team";
}