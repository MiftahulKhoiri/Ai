#!/usr/bin/env python3
# training.py - Script untuk training MiniGPT

import sys
import json
import os
import glob
import time
import random
import math
from typing import List, Tuple, Dict, Any

# Import dari library minigpt
from minigpt import (
    MiniGPT,
    Tokenizer,  # <-- PERUBAHAN: ByteLevelBPETokenizer → Tokenizer
    AdamW,
    WarmupCosineScheduler,
    generate,
    cross_entropy_loss,
    clip_grad_norm,
    argmax,
    Value
)

# ============================================================
# KONFIGURASI TRAINING
# ============================================================
SEQ_LEN = 16
BATCH_SIZE = 2
EPOCHS = 50
PATIENCE = 3
LR = 0.01
WARMUP_STEPS = 30
TOTAL_STEPS = 200
ADDITIONAL_STEPS = 100
MAX_GRAD_NORM = 1.0
D_MODEL = 4
N_HEADS = 2
N_LAYERS = 1
D_FF = 16
MAX_LEN = 32
DROPOUT = 0.1
VOCAB_SIZE = 200

# ============================================================
# FUNGSI MEMORI
# ============================================================
def get_memory_info():
    """Membaca informasi memori dari /proc (untuk Linux/Android)"""
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

def iter_batches(examples: List[List[int]], batch_size: int, shuffle: bool = True):
    if shuffle:
        indices = list(range(len(examples)))
        random.shuffle(indices)
    else:
        indices = list(range(len(examples)))
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [examples[idx] for idx in batch_indices if idx < len(examples)]
        if batch:
            yield batch

def train_batch(model, optimizer, batch, scheduler=None, max_grad_norm=1.0):
    total_loss = 0.0
    n_samples = 0
    
    optimizer.zero_grad()
    
    for seq in batch:
        if len(seq) < 2:
            continue
        
        input_ids = seq[:-1]
        target_ids = seq[1:]
        
        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids, [])
        
        if n_samples == 0:
            total_loss_value = loss
        else:
            total_loss_value = loss
        
        n_samples += 1
    
    if n_samples > 0:
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
                    config=None, total_steps=0, epoch=0, loss=0.0):
    checkpoint = {
        'config': config or {},
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

def load_checkpoint(path, additional_steps=0, warmup_steps=30, min_lr=1e-5):
    with open(path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    # Restore tokenizer
    tokenizer = Tokenizer()  # <-- PERUBAHAN: ByteLevelBPETokenizer → Tokenizer
    if 'vocab' in checkpoint and checkpoint['vocab']:
        vocab = {k: v for k, v in checkpoint['vocab'].items()}
        inv_vocab = {int(k): v for k, v in checkpoint['inv_vocab'].items()}
        merge_order = [tuple(pair) for pair in checkpoint.get('merge_order', [])]
        tokenizer.set_vocab(vocab)
        tokenizer.set_inv_vocab(inv_vocab)
        tokenizer.set_merge_order(merge_order)
    
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', len(tokenizer.get_vocab()))
    d_model = config.get('d_model', D_MODEL)
    n_heads = config.get('n_heads', N_HEADS)
    n_layers = config.get('n_layers', N_LAYERS)
    d_ff = config.get('d_ff', D_FF)
    max_len = config.get('max_len', MAX_LEN)
    dropout = config.get('dropout', DROPOUT)
    
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    if 'params' in checkpoint and checkpoint['params']:
        params = model.parameters()
        for i, p in enumerate(params):
            if i < len(checkpoint['params']):
                p.data = checkpoint['params'][i]
    
    params = model.parameters()
    lr = checkpoint.get('lr', LR)
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)
    
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
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=lr,
        min_lr=min_lr
    )
    scheduler.set_step_num(scheduler_step)
    
    print(f"✅ Checkpoint loaded: step={scheduler_step}, total_steps={total_steps}")
    
    return model, optimizer, scheduler, tokenizer, config, total_steps

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
        print("Penggunaan: python3 training.py <data.json> [checkpoint.json]")
        return

    data_file = sys.argv[1]
    resume_checkpoint = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(data_file):
        print(f"❌ File data '{data_file}' tidak ditemukan.")
        return
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        print(f"❌ File checkpoint '{resume_checkpoint}' tidak ditemukan.")
        return

    print("="*60)
    print("🚀 MINIGPT TRAINING" + (" (RESUME)" if resume_checkpoint else ""))
    print("="*60)

    # 1. Load data
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

    # 2. Tokenizer
    if resume_checkpoint:
        print("\n" + "="*60)
        print("📦 MEMUAT CHECKPOINT")
        print("="*60)
        model, optimizer, scheduler, tokenizer, config, total_steps = load_checkpoint(
            resume_checkpoint,
            additional_steps=ADDITIONAL_STEPS,
            warmup_steps=WARMUP_STEPS,
            min_lr=1e-5
        )
        print(f"  Checkpoint     : {resume_checkpoint}")
        print(f"  Config         : {config}")
        print(f"  Vocab size     : {len(tokenizer.get_vocab())}")
        print(f"  Step terakhir  : {scheduler.get_step_num()}")
        print(f"  Tambah steps   : {ADDITIONAL_STEPS}")
        print(f"  Total steps baru: {total_steps}")
    else:
        print("\n" + "="*60)
        print("🔤 2. MELATIH TOKENIZER BPE")
        print("="*60)
        corpus = "\n".join(sentences)
        print(f"  Total karakter corpus : {len(corpus)}")
        print(f"  Target vocab size     : {VOCAB_SIZE}")
        t0 = time.time()
        tokenizer = Tokenizer()  # <-- PERUBAHAN: ByteLevelBPETokenizer → Tokenizer
        tokenizer.train(corpus, vocab_size=VOCAB_SIZE)
        print(f"  Tokenizer selesai     : {format_time(time.time()-t0)}")
        print(f"  Vocab size aktual     : {len(tokenizer.get_vocab())}")
        vocab_items = list(tokenizer.get_vocab().items())[:10]
        print(f"  10 token pertama      : {[t[0] for t in vocab_items]}")
        print_memory_info("SETELAH TOKENIZER")

    # 3. Encode & build dataset
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
    print(f"  SEQ_LEN             : {SEQ_LEN}")
    print(f"  pad_id              : {pad_id}")
    examples = build_dataset(tokenized, SEQ_LEN, pad_id)
    print(f"  Contoh training     : {len(examples)}")
    if len(examples) == 0:
        print("  ❌ Tidak ada contoh training! Periksa SEQ_LEN dan data.")
        return
    print_memory_info("SETELAH BUILD DATASET")

    # 4. Inisialisasi model
    if not resume_checkpoint:
        print("\n" + "="*60)
        print("🧠 4. INISIALISASI MODEL")
        print("="*60)
        actual_vocab = len(tokenizer.get_vocab())
        est_params = estimate_model_size(actual_vocab, D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_LEN)
        print(f"  Vocab size     : {actual_vocab}")
        print(f"  d_model        : {D_MODEL}")
        print(f"  n_heads        : {N_HEADS}")
        print(f"  n_layers       : {N_LAYERS}")
        print(f"  d_ff           : {D_FF}")
        print(f"  max_len        : {MAX_LEN}")
        print(f"  dropout        : {DROPOUT}")
        print(f"  Est. parameter : {est_params:,}")
        model = MiniGPT(
            vocab_size=actual_vocab,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            max_len=MAX_LEN,
            dropout=DROPOUT
        )
        model.set_training(True)
        all_params = model.parameters()
        print(f"  Parameter aktual: {len(all_params)} tensor")
        print_memory_info("SETELAH MODEL")

        print("\n" + "="*60)
        print("⚙️  5. OPTIMIZER & SCHEDULER")
        print("="*60)
        total_steps = TOTAL_STEPS
        optimizer = AdamW(all_params, lr=LR, weight_decay=0.01)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS,
                                          total_steps=total_steps, base_lr=LR, min_lr=1e-5)
        print(f"  Optimizer      : AdamW")
        print(f"  LR awal        : {LR}")
        print(f"  Weight decay   : 0.01")
        print(f"  Scheduler      : Warmup + Cosine Decay")
        print(f"  Warmup steps   : {WARMUP_STEPS}")
        print(f"  Total steps    : {total_steps}")
        print(f"  Min LR         : 1e-5")
    else:
        print("\n" + "="*60)
        print("🔄 MELANJUTKAN TRAINING")
        print("="*60)
        model.set_training(True)
        print(f"  Model siap, training mode aktif.")
        print(f"  Optimizer & scheduler dari checkpoint.")

    # 5. TRAINING
    print("\n" + "="*60)
    print("🏋️  6. TRAINING")
    print("="*60)
    total_batches = max(1, len(examples) // BATCH_SIZE)
    print(f"  Dataset       : {len(examples)} contoh")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Batch/epoch   : {total_batches}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Max steps     : {total_steps}")
    print(f"  Max grad norm : {MAX_GRAD_NORM}")

    start_time = time.time()
    global_step = scheduler.get_step_num()
    best_loss = float('inf')
    wait = 0
    history = []

    print("\n▶️  MEMULAI TRAINING...\n")
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        print(f"Epoch {epoch}/{EPOCHS}")

        for batch in iter_batches(examples, BATCH_SIZE, shuffle=True):
            if len(batch) == 0:
                continue
            loss, grad_norm = train_batch(model, optimizer, batch, scheduler=scheduler,
                                          max_grad_norm=MAX_GRAD_NORM)
            total_loss += loss
            n_batches += 1
            global_step += 1
            if loss < best_loss:
                best_loss = loss

            mem_mb = get_process_memory_mb()
            mem_str = f"{mem_mb:.0f}MB" if mem_mb > 0 else "N/A"
            print(f"  Batch {n_batches:3d}/{total_batches:<3d} : "
                  f"loss={loss:7.4f}  grad={grad_norm:6.4f}  lr={optimizer.lr:.7f}  RAM={mem_str}")

            if scheduler.get_step_num() >= total_steps:
                break

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, n_batches)
        print(f"  ✅ Epoch {epoch} selesai : "
              f"avg_loss={avg_loss:.4f}  best={best_loss:.4f}  "
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
            print(f"  📈 Loss membaik! Best loss sekarang: {best_loss:.4f}")
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

    # 6. SAVE CHECKPOINT
    print("\n" + "="*60)
    print("💾 7. MENYIMPAN CHECKPOINT BARU")
    print("="*60)
    version = get_next_version()
    checkpoint_path = f"Ai_{version}.json"
    config = {
        "vocab_size": len(tokenizer.get_vocab()),
        "d_model": model.d_model,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_len": model.max_len,
        "dropout": DROPOUT
    }
    save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                    tokenizer=tokenizer, config=config, total_steps=total_steps)
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