import json
import os
import glob
import time
import psutil
from memory_monitor import monitor_memory
from minigpt import MiniGPT, ByteLevelBPETokenizer, AdamW, WarmupCosineScheduler, generate
from minigpt_utils import build_dataset, iter_batches, train_batch, save_checkpoint, load_checkpoint

# ============================================================
# KONFIGURASI TRAINING (dapat disesuaikan)
# ============================================================
DATA_FILE = "data.json"          # file JSON berisi list kalimat latih (string)
SEQ_LEN = 16                     # panjang maksimal urutan token (gunakan 16 untuk data pendek)
BATCH_SIZE = 2                   # jumlah contoh per batch (kecilkan jika OOM atau lambat)
EPOCHS = 10                      # berapa kali seluruh data dilewati (epoch)
LR = 0.01                        # learning rate awal (sebelum warmup/cosine decay)
WARMUP_STEPS = 30                # jumlah langkah pemanasan (linear naik dari 0 ke LR)
TOTAL_STEPS = 200                # total langkah training (cukup untuk dataset kecil)
MAX_GRAD_NORM = 1.0              # batas maksimal norm gradien (gradient clipping)
D_MODEL = 8                      # dimensi embedding (kecilkan agar cepat, 8 untuk uji coba)
N_HEADS = 2                      # jumlah head dalam multi-head attention (harus membagi d_model)
N_LAYERS = 1                     # jumlah layer transformer (1 layer untuk data kecil)
D_FF = 16                        # dimensi feed-forward inner layer
MAX_LEN = 32                     # panjang maksimum posisi (sesuaikan dengan SEQ_LEN)
DROPOUT = 0.1                    # probabilitas dropout (regularisasi)
VOCAB_SIZE = 200                 # ukuran kosakata tokenizer BPE (kecilkan untuk mempercepat)

# ============================================================
# FUNGSI UTILITY
# ============================================================

def get_memory_info():
    """Mengembalikan informasi penggunaan RAM saat ini."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / (1024 * 1024)  # RAM dalam MB
    mem_percent = process.memory_percent()
    
    # RAM sistem
    system_mem = psutil.virtual_memory()
    system_total = system_mem.total / (1024 * 1024 * 1024)  # GB
    system_used = system_mem.used / (1024 * 1024 * 1024)    # GB
    system_percent = system_mem.percent
    
    return {
        'process_mb': mem_mb,
        'process_percent': mem_percent,
        'system_total_gb': system_total,
        'system_used_gb': system_used,
        'system_percent': system_percent
    }

def print_memory_info(label="", show_system=True):
    """Mencetak informasi penggunaan RAM."""
    mem = get_memory_info()
    
    if label:
        print(f"\n{'='*60}")
        print(f"📊 MEMORI: {label}")
        print(f"{'='*60}")
    
    print(f"  🔹 Proses ini    : {mem['process_mb']:.1f} MB ({mem['process_percent']:.1f}% dari total RAM)")
    
    if show_system:
        print(f"  🔹 RAM Sistem    : {mem['system_used_gb']:.1f} GB / {mem['system_total_gb']:.1f} GB "
              f"({mem['system_percent']:.1f}%)")
        print(f"  🔹 RAM Tersisa   : {mem['system_total_gb'] - mem['system_used_gb']:.1f} GB")

def get_next_version():
    """Mencari versi checkpoint tertinggi lalu mengembalikan versi berikutnya."""
    files = glob.glob("Ai_*.json")
    max_ver = 0
    for f in files:
        try:
            ver_str = f.split("Ai_")[1].split(".json")[0]
            ver = int(ver_str)
            if ver > max_ver:
                max_ver = ver
        except:
            pass
    return max_ver + 1

def format_time(seconds):
    """Mengonversi detik ke string jam:menit:detik."""
    if seconds < 60:
        return f"{seconds:.1f} detik"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m} menit {s} detik"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h} jam {m} menit {s} detik"

def estimate_model_size(vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
    """Estimasi jumlah parameter model."""
    # Embedding: vocab_size * d_model + max_len * d_model
    emb_params = vocab_size * d_model + max_len * d_model
    
    # Per Transformer layer:
    # - LayerNorm: 2 * d_model (weight + bias) * 2 (attention + FFN)
    # - Multi-head attention: 4 * d_model * d_model (Q, K, V, O projections)
    # - Feed-forward: 2 * d_model * d_ff + d_ff + d_model (W1, W2, bias)
    attn_params = 4 * d_model * d_model
    ffn_params = 2 * d_model * d_ff + d_ff + d_model
    ln_params = 4 * d_model
    per_layer = attn_params + ffn_params + ln_params
    
    # Final LayerNorm + output projection
    final_params = 2 * d_model + d_model * vocab_size
    
    total = emb_params + n_layers * per_layer + final_params
    return total

# ============================================================
# FUNGSI TEST GENERATE
# ============================================================

def test_generate(model, tokenizer):
    """Tes singkat hasil generate setelah training."""
    prompts = ["Halo", "Apa kabar", "Saya suka"]
    print("\n" + "="*60)
    print("🧪 TES GENERATE")
    print("="*60)
    for i, p in enumerate(prompts, 1):
        result = generate(model, tokenizer, p, max_new_tokens=20)
        print(f"  [{i}] Prompt  : {p}")
        print(f"      Hasil   : {result}")
        print()

# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("="*60)
    print("🚀 MINIGPT TRAINING")
    print("="*60)
    
    # Informasi sistem
    print_memory_info("SEBELUM LOAD DATA")
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n" + "="*60)
    print("📂 1. MEMUAT DATA")
    print("="*60)
    
    monitor_memory()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    
    total_chars = sum(len(s) for s in sentences)
    avg_chars = total_chars / max(1, len(sentences))
    
    print(f"  📄 File           : {DATA_FILE}")
    print(f"  📝 Jumlah kalimat : {len(sentences)}")
    print(f"  📊 Total karakter : {total_chars}")
    print(f"  📏 Rata-rata      : {avg_chars:.1f} karakter/kalimat")
    print_memory_info("SETELAH LOAD DATA", show_system=True)
    
    # ============================================================
    # 2. TRAINING TOKENIZER
    # ============================================================
    print("\n" + "="*60)
    print("🔤 2. MELATIH TOKENIZER BPE")
    print("="*60)
    
    corpus = "\n".join(sentences)
    print(f"  📝 Total karakter corpus: {len(corpus)}")
    print(f"  🎯 Target vocab size    : {VOCAB_SIZE}")
    
    t0 = time.time()
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(corpus, vocab_size=VOCAB_SIZE)
    tokenizer_time = time.time() - t0
    
    actual_vocab = len(tokenizer.vocab)
    print(f"  ✅ Tokenizer selesai    : {format_time(tokenizer_time)}")
    print(f"  📚 Vocab size aktual    : {actual_vocab}")
    print(f"  🔹 Special tokens       : <pad>, <bos>, <eos>, <unk>")
    
    # Tampilkan beberapa token contoh
    vocab_items = list(tokenizer.vocab.items())[:10]
    print(f"  🔹 10 token pertama     : {[t[0] for t in vocab_items]}")
    
    print_memory_info("SETELAH TOKENIZER")
    
    # ============================================================
    # 3. ENCODE DAN BUILD DATASET
    # ============================================================
    print("\n" + "="*60)
    print("🔢 3. ENCODE & BUILD DATASET")
    print("="*60)
    
    pad_id = tokenizer.get_vocab().at('<pad>')
    
    tokenized = []
    total_tokens = 0
    for sent in sentences:
        ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
        tokenized.append(ids)
        total_tokens += len(ids)
    
    avg_tokens = total_tokens / max(1, len(sentences))
    print(f"  🔹 Total token        : {total_tokens}")
    print(f"  🔹 Rata-rata token    : {avg_tokens:.1f} token/kalimat")
    print(f"  🔹 SEQ_LEN            : {SEQ_LEN}")
    print(f"  🔹 pad_id             : {pad_id}")
    
    examples = build_dataset(tokenized, SEQ_LEN, pad_id)
    print(f"  ✅ Contoh training    : {len(examples)}")
    
    if len(examples) == 0:
        print("  ❌ ERROR: Tidak ada contoh training! Periksa SEQ_LEN dan data.")
        return
    
    print_memory_info("SETELAH BUILD DATASET")
    
    # ============================================================
    # 4. INISIALISASI MODEL
    # ============================================================
    print("\n" + "="*60)
    print("🧠 4. INISIALISASI MODEL")
    print("="*60)
    
    est_params = estimate_model_size(actual_vocab, D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_LEN)
    print(f"  🔹 Vocab size    : {actual_vocab}")
    print(f"  🔹 d_model       : {D_MODEL}")
    print(f"  🔹 n_heads       : {N_HEADS}")
    print(f"  🔹 n_layers      : {N_LAYERS}")
    print(f"  🔹 d_ff          : {D_FF}")
    print(f"  🔹 max_len       : {MAX_LEN}")
    print(f"  🔹 dropout       : {DROPOUT}")
    print(f"  📊 Est. parameter: {est_params:,}")
    
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
    print(f"  ✅ Model siap (mode: TRAINING)")
    
    # Hitung parameter aktual
    all_params = model.parameters()
    print(f"  📊 Parameter aktual: {len(all_params)} tensor")
    print_memory_info("SETELAH INISIALISASI MODEL")
    
    # ============================================================
    # 5. INISIALISASI OPTIMIZER & SCHEDULER
    # ============================================================
    print("\n" + "="*60)
    print("⚙️  5. OPTIMIZER & SCHEDULER")
    print("="*60)
    
    optimizer = AdamW(all_params, lr=LR, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS,
                                      total_steps=TOTAL_STEPS, base_lr=LR, min_lr=1e-5)
    
    print(f"  🔹 Optimizer     : AdamW")
    print(f"  🔹 LR awal       : {LR}")
    print(f"  🔹 Weight decay  : 0.01")
    print(f"  🔹 Scheduler     : Warmup + Cosine Decay")
    print(f"  🔹 Warmup steps  : {WARMUP_STEPS}")
    print(f"  🔹 Total steps   : {TOTAL_STEPS}")
    print(f"  🔹 Min LR        : 1e-5")
    
    # ============================================================
    # 6. TRAINING LOOP
    # ============================================================
    print("\n" + "="*60)
    print("🏋️  6. TRAINING")
    print("="*60)
    
    total_batches = max(1, len(examples) // BATCH_SIZE)
    batches_per_epoch = total_batches
    
    print(f"  📊 Dataset      : {len(examples)} contoh")
    print(f"  📦 Batch size   : {BATCH_SIZE}")
    print(f"  📋 Batch/epoch  : {batches_per_epoch}")
    print(f"  🔄 Epochs       : {EPOCHS}")
    print(f"  📊 Total steps  : {EPOCHS * batches_per_epoch} (dibatasi scheduler: {TOTAL_STEPS})")
    print(f"  📏 Max grad norm: {MAX_GRAD_NORM}")
    
    start_time = time.time()
    global_step = 0
    
    # Untuk tracking progress
    best_loss = float('inf')
    history = []
    
    print("\n" + "-"*60)
    print("▶️  MEMULAI TRAINING...")
    print("-"*60)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        
        print(f"\n┌──────────────────────────────────────────────────────────┐")
        print(f"│ EPOCH {epoch}/{EPOCHS}                                          │")
        print(f"├──────────┬────────────┬──────────┬──────────┬──────────────┤")
        print(f"│  Batch   │    Loss    │ Grad Norm│    LR    │   Memori     │")
        print(f"├──────────┼────────────┼──────────┼──────────┼──────────────┤")
        
        for batch in iter_batches(examples, BATCH_SIZE, shuffle=True):
            if len(batch) == 0:
                continue
            
            # Train satu batch
            loss, grad_norm = train_batch(model, optimizer, batch, scheduler=scheduler,
                                          max_grad_norm=MAX_GRAD_NORM)
            
            total_loss += loss
            n_batches += 1
            global_step += 1
            
            # Update best loss
            if loss < best_loss:
                best_loss = loss
            
            # Info memori
            mem = get_memory_info()
            mem_str = f"{mem['process_mb']:.0f} MB"
            
            # Cetak progress setiap batch
            print(f"│ {n_batches:4d}/{batches_per_epoch:<4d} "
                  f"│ {loss:10.4f} "
                  f"│ {grad_norm:8.4f} "
                  f"│ {optimizer.lr:.8f} "
                  f"│ {mem_str:>10s}   │")
            
            # Cek batas total steps
            if scheduler.get_step_num() >= TOTAL_STEPS:
                print(f"├──────────┴────────────┴──────────┴──────────┴──────────────┤")
                print(f"│ ⏰ Batas TOTAL_STEPS ({TOTAL_STEPS}) tercapai.               │")
                print(f"└──────────────────────────────────────────────────────────┘")
                break
        
        # Akhir epoch
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, n_batches)
        
        elapsed_total = time.time() - start_time
        eta = (elapsed_total / global_step) * (TOTAL_STEPS - global_step) if global_step > 0 else 0
        
        print(f"├──────────┴────────────┴──────────┴──────────┴──────────────┤")
        print(f"│ ✅ Epoch {epoch}/{EPOCHS} selesai                                    │")
        print(f"│    Avg Loss   : {avg_loss:.4f}                                  │")
        print(f"│    Best Loss  : {best_loss:.4f}                                  │")
        print(f"│    Waktu      : {format_time(epoch_time)}                          │")
        print(f"│    Step       : {global_step}/{TOTAL_STEPS}                              │")
        print(f"│    ETA        : {format_time(eta)}                          │")
        print(f"└──────────────────────────────────────────────────────────┘")
        
        history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'lr': optimizer.lr,
            'step': global_step,
            'time': format_time(epoch_time)
        })
        
        # Cek batas total steps
        if scheduler.get_step_num() >= TOTAL_STEPS:
            print(f"\n⏰ Training berhenti: mencapai batas {TOTAL_STEPS} steps.")
            break
    
    # ============================================================
    # 7. RINGKASAN TRAINING
    # ============================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("📊 RINGKASAN TRAINING")
    print("="*60)
    print(f"  ⏱️  Total waktu      : {format_time(total_time)}")
    print(f"  📋 Total steps       : {global_step}")
    print(f"  📉 Loss terendah     : {best_loss:.4f}")
    print(f"  📈 LR terakhir       : {optimizer.lr:.8f}")
    print(f"  🔄 Epoch selesai     : {epoch}/{EPOCHS}")
    print(f"  ⏱️  Rata-rata/batch  : {format_time(total_time / max(1, global_step))}")
    
    if len(history) > 0:
        print(f"\n  Riwayat per-epoch:")
        print(f"  {'Epoch':<8} {'Avg Loss':<12} {'Best':<12} {'LR':<12} {'Waktu':<15}")
        print(f"  {'-'*55}")
        for h in history:
            print(f"  {h['epoch']:<8} {h['avg_loss']:<12.4f} {h['best_loss']:<12.4f} "
                  f"{h['lr']:<12.8f} {h['time']:<15}")
    
    print_memory_info("SETELAH TRAINING")
    
    # ============================================================
    # 8. SAVE CHECKPOINT
    # ============================================================
    print("\n" + "="*60)
    print("💾 8. MENYIMPAN CHECKPOINT")
    print("="*60)
    
    version = get_next_version()
    checkpoint_path = f"Ai_{version}.json"
    
    config = {
        "vocab_size": actual_vocab,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_len": MAX_LEN,
        "dropout": DROPOUT
    }
    
    t0 = time.time()
    save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                    tokenizer=tokenizer, config=config)
    save_time = time.time() - t0
    
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    print(f"  💾 Checkpoint      : {checkpoint_path}")
    print(f"  📏 Ukuran file     : {file_size_mb:.2f} MB")
    print(f"  ⏱️  Waktu simpan    : {format_time(save_time)}")
    print(f"  ✅ Model berhasil disimpan!")
    
    # ============================================================
    # 9. TES GENERATE
    # ============================================================
    test_generate(model, tokenizer)
    
    print("\n" + "="*60)
    print("🎉 TRAINING SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()