// bindings.cpp - Tambahan binding untuk fitur baru

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "config.h"
#include "dataloader.h"
#include "sampling.h"
#include "metrics.h"
#include "checkpoint.h"
#include "generation_advanced.h"
#include "visualization.h"
#include "test.h"
#include "server.h"
#include "quantization.h"

namespace py = pybind11;

// Tambahkan di PYBIND11_MODULE:

// Config
py::class_<ModelConfig>(m, "ModelConfig")
    .def(py::init<>())
    .def_readwrite("vocab_size", &ModelConfig::vocab_size)
    .def_readwrite("d_model", &ModelConfig::d_model)
    .def_readwrite("n_heads", &ModelConfig::n_heads)
    .def_readwrite("n_layers", &ModelConfig::n_layers)
    .def_readwrite("d_ff", &ModelConfig::d_ff)
    .def_readwrite("max_len", &ModelConfig::max_len)
    .def_readwrite("dropout", &ModelConfig::dropout)
    .def_readwrite("learning_rate", &ModelConfig::learning_rate)
    .def_readwrite("batch_size", &ModelConfig::batch_size)
    .def_readwrite("epochs", &ModelConfig::epochs)
    .def_readwrite("temperature", &ModelConfig::temperature)
    .def_readwrite("top_k", &ModelConfig::top_k)
    .def_readwrite("top_p", &ModelConfig::top_p)
    .def_static("from_json", &ModelConfig::from_json)
    .def("to_json", &ModelConfig::to_json);

// DataLoader
py::class_<DataLoader>(m, "DataLoader")
    .def(py::init<ByteLevelBPETokenizer&, int, int>())
    .def("load_text", &DataLoader::load_text)
    .def("load_texts", &DataLoader::load_texts)
    .def("next_batch", &DataLoader::next_batch)
    .def("shuffle", &DataLoader::shuffle)
    .def("num_batches", &DataLoader::num_batches)
    .def("reset", &DataLoader::reset)
    .def("size", &DataLoader::size);

// Sampling functions
m.def("greedy_sample", &sampling::greedy_sample);
m.def("top_k_sample", &sampling::top_k_sample);
m.def("top_p_sample", &sampling::top_p_sample);
m.def("batch_sample", &sampling::batch_sample);

// Metrics
m.def("perplexity", &metrics::perplexity);
m.def("accuracy", &metrics::accuracy);
m.def("bleu_score", &metrics::bleu_score);

// TrainingLogger
py::class_<metrics::TrainingLogger>(m, "TrainingLogger")
    .def(py::init<const std::string&>())
    .def("log_epoch", &metrics::TrainingLogger::log_epoch)
    .def("log_metric", &metrics::TrainingLogger::log_metric)
    .def("save_logs", &metrics::TrainingLogger::save_logs)
    .def("print_summary", &metrics::TrainingLogger::print_summary)
    .def("get_losses", &metrics::TrainingLogger::get_losses)
    .def("get_accuracies", &metrics::TrainingLogger::get_accuracies)
    .def("get_lrs", &metrics::TrainingLogger::get_lrs);

// ProgressBar
py::class_<metrics::ProgressBar>(m, "ProgressBar")
    .def(py::init<int, const std::string&>())
    .def("update", &metrics::ProgressBar::update)
    .def("increment", &metrics::ProgressBar::increment)
    .def("finish", &metrics::ProgressBar::finish);

// CheckpointManager
py::class_<CheckpointManager>(m, "CheckpointManager")
    .def(py::init<const std::string&>())
    .def("save", &CheckpointManager::save)
    .def("load", &CheckpointManager::load)
    .def("set_auto_save", &CheckpointManager::set_auto_save)
    .def("set_max_keep", &CheckpointManager::set_max_keep)
    .def("get_latest_checkpoint", &CheckpointManager::get_latest_checkpoint)
    .def("list_checkpoints", &CheckpointManager::list_checkpoints)
    .def("clean_old_checkpoints", &CheckpointManager::clean_old_checkpoints);

// AdvancedGeneration
py::class_<advanced_generation::AdvancedGenerator>(m, "AdvancedGenerator")
    .def(py::init<MiniGPT&, ByteLevelBPETokenizer&>())
    .def("set_config", &advanced_generation::AdvancedGenerator::set_config)
    .def("get_config", &advanced_generation::AdvancedGenerator::get_config)
    .def("generate", &advanced_generation::AdvancedGenerator::generate);

// Visualization
m.def("graph_to_dot", &visualization::graph_to_dot);
m.def("plot_attention", &visualization::plot_attention);
m.def("save_embeddings", &visualization::save_embeddings);

// Quantization
m.def("quantize_float_to_int8", &quantization::quantize_float_to_int8);
m.def("dequantize_int8_to_float", &quantization::dequantize_int8_to_float);
m.def("prune_weights", &quantization::prune_weights);
m.def("compress_model", &quantization::compress_model);
m.def("distill", &quantization::distill);

// Test runner
m.def("run_all_tests", &run_all_tests);