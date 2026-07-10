import json
import os
import glob
import time
from memory_monitor import monitor_memory
from minigpt import MiniGPT, ByteLevelBPETokenizer, AdamW, WarmupCosineScheduler, generate
from minigpt_utils import build_dataset, iter_batches, train_batch, save_checkpoint, load_checkpoint

# ============================================================
# KONFIGURASI TRAINING (dapat disesuaikan)
# ============================================================
DATA_FILE = "data.json"          # file JSON berisi list kalimat latih (string)
SEQ_LEN = 16                     # panjang maksimal urutan token (gunakan 16 untuk data pendek)
BATCH_SIZE = 8                   # jumlah contoh per batch (kecilkan jika OOM atau lambat)
EPOCHS = 2                      # berapa kali seluruh data dilewati (epoch)
LR = 0.01                        # learning rate awal (sebelum warmup/cosine decay)
WARMUP_STEPS = 30                # jumlah langkah pemanasan (linear naik dari 0 ke LR)
TOTAL_STEPS = 200                # total langkah training (cukup untuk dataset kecil)
MAX_GRAD_NORM = 1.0              # batas maksimal norm gradien (gradient clipping)
D_MODEL = 4                      # dimensi embedding (kecilkan agar cepat, 8 untuk uji coba)
N_HEADS = 2                      # jumlah head dalam multi-head attention (harus membagi d_model)
N_LAYERS = 1                     # jumlah layer transformer (1 layer untuk data kecil)
D_FF = 16                        # dimensi feed-forward inner layer
MAX_LEN = 32                     # panjang maksimum posisi (sesuaikan dengan SEQ_LEN)
DROPOUT = 0.1                    # probabilitas dropout (regularisasi)
VOCAB_SIZE = 200                 # ukuran kosakata tokenizer BPE (kecilkan untuk mempercepat)

# ============================================================
# FUNGSI MEMORI (Tanpa psutil, untuk Termux/Android)
# ============================================================

def get_memory_info():
    """Membaca informasi RAM dari /proc (kompatibel dengan Android/Termux)."""
    mem_info = {
        'process_mb': 0.0,
        'system_total_mb': 0.0,
        'system_available_mb': 0.0,
        'system_percent': 0.0
    }

    try:
        # RAM proses (dari /proc/self/status)
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    mem_info['process_mb'] = rss_kb / 1024.0
                    break

        # RAM sistem (dari /proc/meminfo)
        total_kb = 0
        available_kb = 0
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
                elif line.startswith('MemFree:'):
                    if available_kb == 0:
                        available_kb = int(line.split()[1])

        if total_kb > 0:
            mem_info['system_total_mb'] = total_kb / 1024.0
            used_kb = total_kb - available_kb
            mem_info['system_available_mb'] = available_kb / 1024.0
            mem_info['system_percent'] = (used_kb / total_kb) * 100.0

    except Exception:
        pass

    return mem_info

def print_memory_info(label="", show_system=True):
    """Mencetak informasi penggunaan RAM."""
    mem = get_memory_info()

    if label:
        print(f"\n{'='*60}")
        print(f"📊 MEMORI: {label}")
        print(f"{'='*60}")

    if mem['process_mb'] > 0:
        print(f"  🔹 RAM Proses     : {mem['process_mb']:.1f} MB")
    else:
        print(f"  🔹 RAM Proses     : tidak dapat dibaca")

    if show_system and mem['system_total_mb'] > 0:
        total_gb = mem['system_total_mb'] / 1024.0
        avail_gb = mem['system_available_mb'] / 1024.0
        used_gb = total_gb - avail_gb

        print(f"  🔹 RAM Sistem     : {used_gb:.1f} GB / {total_gb:.1f} GB ({mem['system_percent']:.1f}%)")
        print(f"  🔹 RAM Tersedia   : {avail_gb:.1f} GB")
    elif show_system:
        print(f"  🔹 RAM Sistem     : tidak dapat dibaca")

def get_process_memory_mb():
    """Mengembalikan RAM proses dalam MB (untuk ditampilkan di progress)."""
    mem = get_memory_info()
    return mem['process_mb']

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
    """Mengonversi detik ke string yang mudah dibaca."""
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
    emb_params = vocab_size * d_model + max_len * d_model
    attn_params = 4 * d_model * d_model
    ffn_params = 2 * d_model * d_ff + d_ff + d_model
    ln_params = 4 * d_model
    per_layer = attn_params + ffn_params + ln_params
    final_params = 2 * d_model + d_model * vocab_size
    total = emb_params + n_layers * per_layer + final_params
    return total

# ============================================================
# TES GENERATE
# ============================================================

def test_generate(model, tokenizer):
    """Tes singkat hasil generate setelah training."""
    prompts = ["Halo", "Apa kabar", "Saya suka"]
    print("\n" + "="*60)
    print("🧪 TES GENERATE")
    print("="*60)
    for i, p in enumerate(prompts, 1):
        try:
            result = generate(model, tokenizer, p, max_new_tokens=20)
            print(f"  [{i}] Prompt  : {p}")
            print(f"      Hasil   : {result}")
        except Exception as e:
            print(f"  [{i}] Prompt  : {p}")
            print(f"      ❌ Error : {e}")
        print()

# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("="*60)
    print("🚀 MINIGPT TRAINING")
    print("="*60)

    print_memory_info("SEBELUM LOAD DATA")

    # ============================================================
    # 1. LOAD DATA (tampilan sederhana)
    # ============================================================
    print("\n" + "="*60)
    print("📂 1. MEMUAT DATA")
    print("="*60)

    monitor_memory()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    total_chars = sum(len(s) for s in sentences)
    avg_chars = total_chars / max(1, len(sentences))

    print(f"  File           : {DATA_FILE}")
    print(f"  Jumlah kalimat : {len(sentences)}")
    print(f"  Total karakter : {total_chars}")
    print(f"  Rata-rata      : {avg_chars:.1f} karakter/kalimat")
    print_memory_info("SETELAH LOAD DATA")

    # ============================================================
    # 2. TRAINING TOKENIZER
    # ============================================================
    print("\n" + "="*60)
    print("🔤 2. MELATIH TOKENIZER BPE")
    print("="*60)

    corpus = "\n".join(sentences)
    print(f"  Total karakter corpus : {len(corpus)}")
    print(f"  Target vocab size     : {VOCAB_SIZE}")

    t0 = time.time()
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(corpus, vocab_size=VOCAB_SIZE)
    tokenizer_time = time.time() - t0

    actual_vocab = len(tokenizer.vocab)
    print(f"  Tokenizer selesai     : {format_time(tokenizer_time)}")
    print(f"  Vocab size aktual     : {actual_vocab}")
    print(f"  Special tokens        : <pad>, <bos>, <eos>, <unk>")

    vocab_items = list(tokenizer.vocab.items())[:10]
    print(f"  10 token pertama      : {[t[0] for t in vocab_items]}")

    print_memory_info("SETELAH TOKENIZER")

    # ============================================================
    # 3. ENCODE DAN BUILD DATASET
    # ============================================================
    print("\n" + "="*60)
    print("🔢 3. ENCODE & BUILD DATASET")
    print("="*60)

    pad_id = tokenizer.vocab['<pad>']

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
    print(f"  Model siap (mode: TRAINING)")

    all_params = model.parameters()
    print(f"  Parameter aktual: {len(all_params)} tensor")
    print_memory_info("SETELAH MODEL")

    # ============================================================
    # 5. OPTIMIZER & SCHEDULER
    # ============================================================
    print("\n" + "="*60)
    print("⚙️  5. OPTIMIZER & SCHEDULER")
    print("="*60)

    optimizer = AdamW(all_params, lr=LR, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS,
                                      total_steps=TOTAL_STEPS, base_lr=LR, min_lr=1e-5)

    print(f"  Optimizer      : AdamW")
    print(f"  LR awal        : {LR}")
    print(f"  Weight decay   : 0.01")
    print(f"  Scheduler      : Warmup + Cosine Decay")
    print(f"  Warmup steps   : {WARMUP_STEPS}")
    print(f"  Total steps    : {TOTAL_STEPS}")
    print(f"  Min LR         : 1e-5")

    # ============================================================
    # 6. TRAINING (tampilan ringkas & langsung)
    # ============================================================
    print("\n" + "="*60)
    print("🏋️  6. TRAINING")
    print("="*60)

    total_batches = max(1, len(examples) // BATCH_SIZE)

    print(f"  Dataset       : {len(examples)} contoh")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Batch/epoch   : {total_batches}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Max steps     : {TOTAL_STEPS}")
    print(f"  Max grad norm : {MAX_GRAD_NORM}")

    start_time = time.time()
    global_step = 0
    best_loss = float('inf')
    history = []

    print("\n▶️  MEMULAI TRAINING...\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        # Print header epoch
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

            # Info RAM
            mem_mb = get_process_memory_mb()
            mem_str = f"{mem_mb:.0f}MB" if mem_mb > 0 else "N/A"

            # Cetak per batch (setiap batch)
            print(f"  Batch {n_batches:3d}/{total_batches:<3d} : "
                  f"loss={loss:7.4f}  grad={grad_norm:6.4f}  lr={optimizer.lr:.7f}  RAM={mem_str}")

            if scheduler.step_num >= TOTAL_STEPS:
                break

        # Ringkasan epoch
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, n_batches)
        elapsed_total = time.time() - start_time
        eta = (elapsed_total / global_step) * (TOTAL_STEPS - global_step) if global_step > 0 else 0

        print(f"  ✅ Epoch {epoch} selesai : "
              f"avg_loss={avg_loss:.4f}  best={best_loss:.4f}  "
              f"time={format_time(epoch_time)}  step={global_step}/{TOTAL_STEPS}")
        if eta > 0 and global_step < TOTAL_STEPS:
            print(f"     ETA = {format_time(eta)}")
        print()  # baris kosong antar epoch

        history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'lr': optimizer.lr,
            'step': global_step,
            'time': format_time(epoch_time)
        })

        if scheduler.step_num >= TOTAL_STEPS:
            print(f"⏰ Training berhenti: mencapai batas {TOTAL_STEPS} steps.")
            break

    # ============================================================
    # 7. RINGKASAN (lebih ringkas)
    # ============================================================
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("📊 RINGKASAN TRAINING")
    print("="*60)
    print(f"  Total waktu      : {format_time(total_time)}")
    print(f"  Total steps      : {global_step}")
    print(f"  Best loss        : {best_loss:.4f}")
    print(f"  Last LR          : {optimizer.lr:.8f}")
    print(f"  Epoch terakhir   : {epoch}/{EPOCHS}")
    print(f"  Rata-rata/step   : {format_time(total_time / max(1, global_step))}")

    if len(history) > 0:
        print("\n  Riwayat per-epoch:")
        print("  Epoch  Avg Loss   Best Loss   LR           Waktu")
        print("  " + "-"*50)
        for h in history:
            print(f"  {h['epoch']:3d}    {h['avg_loss']:.4f}     {h['best_loss']:.4f}   {h['lr']:.8f}   {h['time']}")

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

    save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                    tokenizer=tokenizer, config=config)

    if os.path.exists(checkpoint_path):
        file_size_kb = os.path.getsize(checkpoint_path) / 1024.0
        file_size_mb = file_size_kb / 1024.0
        print(f"  Checkpoint    : {checkpoint_path}")
        if file_size_mb >= 1.0:
            print(f"  Ukuran file   : {file_size_mb:.2f} MB")
        else:
            print(f"  Ukuran file   : {file_size_kb:.1f} KB")
        print(f"  Model berhasil disimpan!")
    else:
        print(f"  Gagal menyimpan checkpoint!")

    # ============================================================
    # 9. TES GENERATE
    # ============================================================
    test_generate(model, tokenizer)

    print("\n" + "="*60)
    print("🎉 TRAINING SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()