#!/usr/bin/env python3
# training.py - Script untuk training MiniGPT (versi mmap_ninja, config terpusat)

import sys
import json
import os
import glob
import time
import random
import math
from typing import List, Tuple, Dict, Any

from minigpt import (
    MiniGPT,
    Tokenizer,
    AdamW,
    WarmupCosineScheduler,
    ModelConfig,        # <-- BARU: sumber kebenaran tunggal untuk hyperparameter
    generate,
    cross_entropy_loss,
    clip_grad_norm,
    argmax,
    Value,
    MMapDataset,
    build_mmap_dataset,
)

# ============================================================
# KONFIGURASI TRAINING
# ============================================================
# FIX (unifikasi config): field yang JUGA ada di C++ ModelConfig
# (arsitektur model + hyperparameter optimizer/scheduler) sekarang
# hidup di satu objek "cfg", persis sama strukturnya dengan yang
# dipakai train.cpp/main.cpp lewat bindings.cpp. Field yang MURNI
# milik script Python ini (tidak relevan untuk C++/checkpoint biner)
# tetap jadi konstanta terpisah di bawah.
def build_default_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.vocab_size = 200
    cfg.d_model = 16
    cfg.n_heads = 4
    cfg.n_layers = 2
    cfg.d_ff = 16
    cfg.max_len = 32
    cfg.dropout = 0.1
    cfg.learning_rate = 0.01
    cfg.weight_decay = 0.01
    cfg.warmup_steps = 30
    cfg.total_steps = 200
    cfg.epochs = 50
    cfg.batch_size = 16
    return cfg

# Hyperparameter yang TIDAK ada di ModelConfig (spesifik alur Python ini)
PATIENCE = 3
MAX_GRAD_NORM = 1.0
ADDITIONAL_STEPS = 100

# ============================================================
# FUNGSI MEMORI
# ============================================================
def get_memory_info():
    mem_info = {
        'process_mb': 0.0,
        'system_total_mb': 0.0,
        'system_available_mb': 0.0,
        'system_percent': 0.0
    }
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    mem_info['process_mb'] = int(line.split()[1]) / 1024.0
                    break
        total_kb = available_kb = 0
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
                elif line.startswith('MemFree:') and available_kb == 0:
                    available_kb = int(line.split()[1])
        if total_kb > 0:
            mem_info['system_total_mb'] = total_kb / 1024.0
            mem_info['system_available_mb'] = available_kb / 1024.0
            mem_info['system_percent'] = ((total_kb - available_kb) / total_kb) * 100.0
    except:
        pass
    return mem_info

def print_memory_info(label="", show_system=True):
    mem = get_memory_info()
    if label:
        print(f"\n{'='*60}\n📊 MEMORI: {label}\n{'='*60}")
    print(f"  🔹 RAM Proses     : {mem['process_mb']:.1f} MB" if mem['process_mb'] > 0 else "  🔹 RAM Proses     : tidak dapat dibaca")
    if show_system and mem['system_total_mb'] > 0:
        total_gb = mem['system_total_mb'] / 1024
        avail_gb = mem['system_available_mb'] / 1024
        used_gb = total_gb - avail_gb
        print(f"  🔹 RAM Sistem     : {used_gb:.1f} GB / {total_gb:.1f} GB ({mem['system_percent']:.1f}%)")
        print(f"  🔹 RAM Tersedia   : {avail_gb:.1f} GB")
    elif show_system:
        print(f"  🔹 RAM Sistem     : tidak dapat dibaca")

def get_process_memory_mb():
    return get_memory_info()['process_mb']

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} detik"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m} menit {s} detik"
    else:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h} jam {m} menit {s} detik"

def get_next_version():
    files = glob.glob("Ai_*.json")
    max_ver = 0
    for f in files:
        try:
            ver = int(f.split("Ai_")[1].split(".json")[0])
            if ver > max_ver:
                max_ver = ver
        except:
            pass
    return max_ver + 1

def estimate_model_size(vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
    emb = vocab_size * d_model + max_len * d_model
    attn = 4 * d_model * d_model
    ffn = 2 * d_model * d_ff + d_ff + d_model
    ln = 4 * d_model
    return emb + n_layers * (attn + ffn + ln) + 2 * d_model + d_model * vocab_size

# ============================================================
# FUNGSI DATASET
# ============================================================
def build_dataset(tokenized: List[List[int]], seq_len: int, pad_id: int) -> List[List[int]]:
    examples = []
    for tokens in tokenized:
        if len(tokens) < 2:
            continue
        if len(tokens) < seq_len:
            padded = tokens + [pad_id] * (seq_len - len(tokens))
            examples.append(padded)
        else:
            for i in range(0, len(tokens) - seq_len + 1, seq_len // 2):
                seq = tokens[i:i + seq_len]
                if len(seq) == seq_len:
                    examples.append(seq)
    return examples

def iter_batches_mmap(dataset, batch_size: int, shuffle: bool = True):
    n = dataset.size()
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, n, batch_size):
        batch_indices = indices[i:i + batch_size]
        if batch_indices:
            yield dataset.get_batch(batch_indices)

def train_batch(model, optimizer, batch, scheduler=None, max_grad_norm=1.0):
    total_loss_value = None
    n_samples = 0

    optimizer.zero_grad()

    for seq in batch:
        if len(seq) < 2:
            continue

        input_ids = seq[:-1]
        target_ids = seq[1:]

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids, [])

        if total_loss_value is None:
            total_loss_value = loss
        else:
            total_loss_value = total_loss_value + loss

        n_samples += 1

    if n_samples > 0:
        total_loss_value = total_loss_value / n_samples
        total_loss_value.backward()
        params = model.parameters()
        grad_norm = clip_grad_norm(params, max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        return total_loss_value.data, grad_norm

    return 0.0, 0.0

# ============================================================
# FUNGSI SAVE/LOAD CHECKPOINT
# ============================================================
def save_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None,
                    cfg: ModelConfig = None, total_steps=0, epoch=0, loss=0.0):
    checkpoint = {
        # FIX (unifikasi config): config disimpan langsung dari field cfg,
        # bukan dict manual terpisah yang bisa lupa disinkronkan.
        'config': {
            'vocab_size': cfg.vocab_size,
            'd_model': cfg.d_model,
            'n_heads': cfg.n_heads,
            'n_layers': cfg.n_layers,
            'd_ff': cfg.d_ff,
            'max_len': cfg.max_len,
            'dropout': cfg.dropout,
            'learning_rate': cfg.learning_rate,
            'weight_decay': cfg.weight_decay,
            'warmup_steps': cfg.warmup_steps,
            'batch_size': cfg.batch_size,
            'epochs': cfg.epochs,
        } if cfg else {},
        'total_steps': total_steps,
        'epoch': epoch,
        'loss': loss,
        'vocab': tokenizer.get_vocab() if tokenizer else {},
        'inv_vocab': {str(k): v for k, v in tokenizer.get_inv_vocab().items()} if tokenizer else {},
        'merge_order': tokenizer.get_merge_order() if tokenizer else [],
        'params': [],
        'm': [],
        'v': [],
        't': 0,
        'lr': 0
    }

    if model:
        params = model.parameters()
        checkpoint['params'] = [p.data for p in params]

    if optimizer:
        checkpoint['m'] = optimizer.get_m()
        checkpoint['v'] = optimizer.get_v()
        checkpoint['t'] = optimizer.get_t()
        checkpoint['lr'] = optimizer.lr

    if scheduler:
        checkpoint['scheduler_step'] = scheduler.get_step_num()

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def load_checkpoint(path, additional_steps=0, min_lr=1e-5):
    with open(path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)

    tokenizer = Tokenizer()
    if 'vocab' in checkpoint and checkpoint['vocab']:
        vocab = {k: v for k, v in checkpoint['vocab'].items()}
        inv_vocab = {int(k): v for k, v in checkpoint['inv_vocab'].items()}
        merge_order = [tuple(pair) for pair in checkpoint.get('merge_order', [])]
        tokenizer.set_vocab(vocab)
        tokenizer.set_inv_vocab(inv_vocab)
        tokenizer.set_merge_order(merge_order)

    # FIX (unifikasi config): rekonstruksi cfg dari checkpoint, dengan
    # default ModelConfig() sebagai fallback per-field kalau checkpoint
    # lama tidak punya field tertentu (mis. checkpoint dari sebelum
    # unifikasi ini dibuat).
    saved = checkpoint.get('config', {})
    cfg = build_default_config()
    cfg.vocab_size = saved.get('vocab_size', len(tokenizer.get_vocab()))
    cfg.d_model = saved.get('d_model', cfg.d_model)
    cfg.n_heads = saved.get('n_heads', cfg.n_heads)
    cfg.n_layers = saved.get('n_layers', cfg.n_layers)
    cfg.d_ff = saved.get('d_ff', cfg.d_ff)
    cfg.max_len = saved.get('max_len', cfg.max_len)
    cfg.dropout = saved.get('dropout', cfg.dropout)
    cfg.learning_rate = checkpoint.get('lr', saved.get('learning_rate', cfg.learning_rate))
    cfg.weight_decay = saved.get('weight_decay', cfg.weight_decay)
    cfg.warmup_steps = saved.get('warmup_steps', cfg.warmup_steps)
    cfg.batch_size = saved.get('batch_size', cfg.batch_size)
    cfg.epochs = saved.get('epochs', cfg.epochs)

    model = MiniGPT(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
        dropout=cfg.dropout
    )

    if 'params' in checkpoint and checkpoint['params']:
        params = model.parameters()
        for i, p in enumerate(params):
            if i < len(checkpoint['params']):
                p.data = checkpoint['params'][i]

    params = model.parameters()
    optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    if 'm' in checkpoint and checkpoint['m']:
        optimizer.set_m(checkpoint['m'])
    if 'v' in checkpoint and checkpoint['v']:
        optimizer.set_v(checkpoint['v'])
    if 't' in checkpoint:
        optimizer.set_t(checkpoint['t'])
    if 'lr' in checkpoint:
        optimizer.lr = checkpoint['lr']

    scheduler_step = checkpoint.get('scheduler_step', 0)
    total_steps = scheduler_step + additional_steps

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_steps,
        base_lr=cfg.learning_rate,
        min_lr=min_lr
    )
    scheduler.set_step_num(scheduler_step)

    print(f"✅ Checkpoint loaded: step={scheduler_step}, total_steps={total_steps}")

    return model, optimizer, scheduler, tokenizer, cfg, total_steps

def test_generate(model, tokenizer):
    prompts = ["Halo", "Apa kabar", "Saya suka"]
    print("\n" + "="*60 + "\n🧪 TES GENERATE\n" + "="*60)
    for i, p in enumerate(prompts, 1):
        try:
            result = generate(model, tokenizer, p, max_tokens=20)
            if isinstance(result, list):
                result = result[0] if result else ""
            print(f"  [{i}] Prompt  : {p}\n      Hasil   : {result}")
        except Exception as e:
            print(f"  [{i}] Prompt  : {p}\n      ❌ Error : {e}")
        print()

# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python3 training.py <data.json> [checkpoint.json] [config.json]")
        return

    data_file = sys.argv[1]
    resume_checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
    config_file = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(data_file):
        print(f"❌ File data '{data_file}' tidak ditemukan.")
        return
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        print(f"❌ File checkpoint '{resume_checkpoint}' tidak ditemukan.")
        return

    print("="*60)
    print("🚀 MINIGPT TRAINING" + (" (RESUME)" if resume_checkpoint else ""))
    print("="*60)

    print_memory_info("SEBELUM LOAD DATA")
    print("\n" + "="*60)
    print("📂 1. MEMUAT DATA")
    print("="*60)
    with open(data_file, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    total_chars = sum(len(s) for s in sentences)
    avg_chars = total_chars / max(1, len(sentences))
    print(f"  File           : {data_file}")
    print(f"  Jumlah kalimat : {len(sentences)}")
    print(f"  Total karakter : {total_chars}")
    print(f"  Rata-rata      : {avg_chars:.1f} karakter/kalimat")
    print_memory_info("SETELAH LOAD DATA")

    if resume_checkpoint:
        print("\n" + "="*60)
        print("📦 MEMUAT CHECKPOINT")
        print("="*60)
        model, optimizer, scheduler, tokenizer, cfg, total_steps = load_checkpoint(
            resume_checkpoint,
            additional_steps=ADDITIONAL_STEPS,
            min_lr=1e-5
        )
        print(f"  Checkpoint     : {resume_checkpoint}")
        print(f"  Vocab size     : {len(tokenizer.get_vocab())}")
        print(f"  Step terakhir  : {scheduler.get_step_num()}")
        print(f"  Tambah steps   : {ADDITIONAL_STEPS}")
        print(f"  Total steps baru: {total_steps}")
    else:
        # FIX (unifikasi config): cfg dibangun dari file JSON kalau
        # diberikan (kompatibel dengan yang ditulis main.cpp lewat
        # cfg.to_json / --config), kalau tidak pakai default script ini.
        if config_file and os.path.exists(config_file):
            cfg = ModelConfig.from_json(config_file)
            print(f"  Config dimuat dari: {config_file}")
        else:
            cfg = build_default_config()
            print("  Menggunakan ModelConfig default (script)")

        print("\n" + "="*60)
        print("🔤 2. MELATIH TOKENIZER BPE")
        print("="*60)
        corpus = "\n".join(sentences)
        print(f"  Total karakter corpus : {len(corpus)}")
        print(f"  Target vocab size     : {cfg.vocab_size}")
        t0 = time.time()
        tokenizer = Tokenizer()
        tokenizer.train(corpus, vocab_size=cfg.vocab_size)
        print(f"  Tokenizer selesai     : {format_time(time.time()-t0)}")
        print(f"  Vocab size aktual     : {len(tokenizer.get_vocab())}")
        vocab_items = list(tokenizer.get_vocab().items())[:10]
        print(f"  10 token pertama      : {[t[0] for t in vocab_items]}")
        print_memory_info("SETELAH TOKENIZER")

    print("\n" + "="*60)
    print("🔢 3. ENCODE & BUILD DATASET")
    print("="*60)
    vocab = tokenizer.get_vocab()
    pad_id = vocab.get('<pad>', 0)
    tokenized = []
    total_tokens = 0
    for sent in sentences:
        ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
        tokenized.append(ids)
        total_tokens += len(ids)
    avg_tokens = total_tokens / max(1, len(sentences))
    print(f"  Total token         : {total_tokens}")
    print(f"  Rata-rata token     : {avg_tokens:.1f} token/kalimat")
    print(f"  max_len (SEQ_LEN)   : {cfg.max_len}")
    print(f"  pad_id              : {pad_id}")

    examples = build_dataset(tokenized, cfg.max_len, pad_id)
    n_examples = len(examples)
    print(f"  Contoh training     : {n_examples}")
    if n_examples == 0:
        print("  ❌ Tidak ada contoh training! Periksa max_len dan data.")
        return
    print_memory_info("SETELAH BUILD DATASET")

    print("\n" + "="*60)
    print("💽 3b. MENULIS DATASET KE MMAP FILE")
    print("="*60)
    mmap_path = os.path.splitext(os.path.basename(data_file))[0] + f".seq{cfg.max_len}.mmpn"
    t0 = time.time()
    build_mmap_dataset(mmap_path, examples, cfg.max_len)
    print(f"  File mmap      : {mmap_path}")
    print(f"  Waktu tulis    : {format_time(time.time()-t0)}")
    size_mb = os.path.getsize(mmap_path) / (1024*1024)
    print(f"  Ukuran file    : {size_mb:.2f} MB")

    del examples
    dataset = MMapDataset(mmap_path)
    print(f"  Dataset dimuat : {dataset.size()} contoh, seq_len={dataset.seq_len()} (via mmap)")
    print_memory_info("SETELAH BUILD MMAP DATASET")

    if not resume_checkpoint:
        print("\n" + "="*60)
        print("🧠 4. INISIALISASI MODEL")
        print("="*60)
        cfg.vocab_size = len(tokenizer.get_vocab())  # vocab aktual menang atas target
        est_params = estimate_model_size(cfg.vocab_size, cfg.d_model, cfg.n_layers,
                                          cfg.n_heads, cfg.d_ff, cfg.max_len)
        print(f"  Vocab size     : {cfg.vocab_size}")
        print(f"  d_model        : {cfg.d_model}")
        print(f"  n_heads        : {cfg.n_heads}")
        print(f"  n_layers       : {cfg.n_layers}")
        print(f"  d_ff           : {cfg.d_ff}")
        print(f"  max_len        : {cfg.max_len}")
        print(f"  dropout        : {cfg.dropout}")
        print(f"  Est. parameter : {est_params:,}")
        model = MiniGPT(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_len=cfg.max_len,
            dropout=cfg.dropout
        )
        model.set_training(True)
        all_params = model.parameters()
        print(f"  Parameter aktual: {len(all_params)} tensor")
        print_memory_info("SETELAH MODEL")

        print("\n" + "="*60)
        print("⚙️  5. OPTIMIZER & SCHEDULER")
        print("="*60)
        total_steps = cfg.total_steps
        optimizer = AdamW(all_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=cfg.warmup_steps,
                                          total_steps=total_steps, base_lr=cfg.learning_rate, min_lr=1e-5)
        print(f"  Optimizer      : AdamW")
        print(f"  LR awal        : {cfg.learning_rate}")
        print(f"  Weight decay   : {cfg.weight_decay}")
        print(f"  Warmup steps   : {cfg.warmup_steps}")
        print(f"  Total steps    : {total_steps}")
    else:
        print("\n" + "="*60)
        print("🔄 MELANJUTKAN TRAINING")
        print("="*60)
        model.set_training(True)
        print(f"  Model siap, training mode aktif.")

    print("\n" + "="*60)
    print("🏋️  6. TRAINING")
    print("="*60)
    total_batches = max(1, dataset.size() // cfg.batch_size)
    print(f"  Dataset       : {dataset.size()} contoh (mmap: {mmap_path})")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Batch/epoch   : {total_batches}")
    print(f"  Epochs        : {cfg.epochs}")
    print(f"  Max steps     : {total_steps}")
    print(f"  Max grad norm : {MAX_GRAD_NORM}")

    start_time = time.time()
    global_step = scheduler.get_step_num()
    best_batch_loss = float('inf')
    best_loss = float('inf')
    wait = 0
    history = []

    print("\n▶️  MEMULAI TRAINING...\n")
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        print(f"Epoch {epoch}/{cfg.epochs}")

        for batch in iter_batches_mmap(dataset, cfg.batch_size, shuffle=True):
            if len(batch) == 0:
                continue
            loss, grad_norm = train_batch(model, optimizer, batch, scheduler=scheduler,
                                          max_grad_norm=MAX_GRAD_NORM)
            total_loss += loss
            n_batches += 1
            global_step += 1
            if loss < best_batch_loss:
                best_batch_loss = loss

            mem_mb = get_process_memory_mb()
            mem_str = f"{mem_mb:.0f}MB" if mem_mb > 0 else "N/A"
            print(f"  Batch {n_batches:3d}/{total_batches:<3d} : "
                  f"loss={loss:7.4f}  grad={grad_norm:6.4f}  lr={optimizer.lr:.7f}  RAM={mem_str}")

            if scheduler.get_step_num() >= total_steps:
                break

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, n_batches)
        print(f"  ✅ Epoch {epoch} selesai : "
              f"avg_loss={avg_loss:.4f}  best_batch={best_batch_loss:.4f}  "
              f"time={format_time(epoch_time)}  step={global_step}/{total_steps}")
        print()

        history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'lr': optimizer.lr,
            'step': global_step,
            'time': format_time(epoch_time)
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            print(f"  📈 Loss membaik! Best avg loss sekarang: {best_loss:.4f}")
        else:
            wait += 1
            print(f"  ⏳ Early Stopping: {wait}/{PATIENCE} (loss tidak membaik)")

        if wait >= PATIENCE:
            print(f"🛑 Early stopping dipicu setelah {PATIENCE} epoch tanpa perbaikan!")
            break

        if scheduler.get_step_num() >= total_steps:
            print(f"⏰ Training berhenti: mencapai batas {total_steps} steps.")
            break

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 RINGKASAN TRAINING")
    print("="*60)
    print(f"  Total waktu      : {format_time(total_time)}")
    print(f"  Total steps      : {global_step}")
    print(f"  Best loss        : {best_loss:.4f}")
    print(f"  Last LR          : {optimizer.lr:.8f}")
    if history:
        print("\n  Riwayat per-epoch:")
        print("  Epoch  Avg Loss   Best Loss   LR           Waktu")
        print("  " + "-"*50)
        for h in history:
            print(f"  {h['epoch']:3d}    {h['avg_loss']:.4f}     {h['best_loss']:.4f}   {h['lr']:.8f}   {h['time']}")
    print_memory_info("SETELAH TRAINING")

    print("\n" + "="*60)
    print("💾 7. MENYIMPAN CHECKPOINT BARU")
    print("="*60)
    version = get_next_version()
    checkpoint_path = f"Ai_{version}.json"
    cfg.epochs = epoch  # catat epoch aktual yang tercapai (kalau early-stop)
    save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                    tokenizer=tokenizer, cfg=cfg, total_steps=total_steps)
    if os.path.exists(checkpoint_path):
        size_kb = os.path.getsize(checkpoint_path) / 1024
        print(f"  Checkpoint    : {checkpoint_path}")
        print(f"  Ukuran file   : {size_kb:.1f} KB" if size_kb < 1024 else f"  Ukuran file   : {size_kb/1024:.2f} MB")
        print(f"  Model berhasil disimpan!")
    else:
        print(f"  Gagal menyimpan checkpoint!")

    test_generate(model, tokenizer)
    print("\n" + "="*60)
    print("🎉 TRAINING SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()