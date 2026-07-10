import sys
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
SEQ_LEN = 16
BATCH_SIZE = 8
EPOCHS = 2
LR = 0.01
WARMUP_STEPS = 30
TOTAL_STEPS = 200          # digunakan saat training baru
ADDITIONAL_STEPS = 100     # jika resume, tambah steps sebanyak ini
MAX_GRAD_NORM = 1.0
D_MODEL = 4
N_HEADS = 2
N_LAYERS = 1
D_FF = 16
MAX_LEN = 32
DROPOUT = 0.1
VOCAB_SIZE = 200

# ============================================================
# FUNGSI MEMORI (tanpa psutil, untuk Termux/Android)
# ============================================================
def get_memory_info():
    mem_info = {'process_mb': 0.0, 'system_total_mb': 0.0, 'system_available_mb': 0.0, 'system_percent': 0.0}
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    mem_info['process_mb'] = int(line.split()[1]) / 1024.0
                    break
        total_kb = available_kb = 0
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'): total_kb = int(line.split()[1])
                elif line.startswith('MemAvailable:'): available_kb = int(line.split()[1])
                elif line.startswith('MemFree:') and available_kb == 0: available_kb = int(line.split()[1])
        if total_kb > 0:
            mem_info['system_total_mb'] = total_kb / 1024.0
            mem_info['system_available_mb'] = available_kb / 1024.0
            mem_info['system_percent'] = ((total_kb - available_kb) / total_kb) * 100.0
    except: pass
    return mem_info

def print_memory_info(label="", show_system=True):
    mem = get_memory_info()
    if label: print(f"\n{'='*60}\n📊 MEMORI: {label}\n{'='*60}")
    print(f"  🔹 RAM Proses     : {mem['process_mb']:.1f} MB" if mem['process_mb'] > 0 else "  🔹 RAM Proses     : tidak dapat dibaca")
    if show_system and mem['system_total_mb'] > 0:
        total_gb = mem['system_total_mb'] / 1024
        avail_gb = mem['system_available_mb'] / 1024
        used_gb = total_gb - avail_gb
        print(f"  🔹 RAM Sistem     : {used_gb:.1f} GB / {total_gb:.1f} GB ({mem['system_percent']:.1f}%)")
        print(f"  🔹 RAM Tersedia   : {avail_gb:.1f} GB")
    elif show_system: print(f"  🔹 RAM Sistem     : tidak dapat dibaca")

def get_process_memory_mb():
    return get_memory_info()['process_mb']

def get_next_version():
    files = glob.glob("Ai_*.json")
    max_ver = 0
    for f in files:
        try:
            ver = int(f.split("Ai_")[1].split(".json")[0])
            if ver > max_ver: max_ver = ver
        except: pass
    return max_ver + 1

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f} detik"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m} menit {s} detik"
    else:
        h, r = divmod(int(seconds), 3600)
        m, s = divmod(r, 60)
        return f"{h} jam {m} menit {s} detik"

def estimate_model_size(vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
    emb = vocab_size * d_model + max_len * d_model
    attn = 4 * d_model * d_model
    ffn = 2 * d_model * d_ff + d_ff + d_model
    ln = 4 * d_model
    return emb + n_layers * (attn + ffn + ln) + 2 * d_model + d_model * vocab_size

def test_generate(model, tokenizer):
    prompts = ["Halo", "Apa kabar", "Saya suka"]
    print("\n" + "="*60 + "\n🧪 TES GENERATE\n" + "="*60)
    for i, p in enumerate(prompts, 1):
        try:
            result = generate(model, tokenizer, p, max_new_tokens=20)
            print(f"  [{i}] Prompt  : {p}\n      Hasil   : {result}")
        except Exception as e:
            print(f"  [{i}] Prompt  : {p}\n      ❌ Error : {e}")
        print()

# ============================================================
# MAIN
# ============================================================
def main():
    # --- Parsing command line ---
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
    monitor_memory()
    with open(data_file, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    total_chars = sum(len(s) for s in sentences)
    avg_chars = total_chars / max(1, len(sentences))
    print(f"  File           : {data_file}")
    print(f"  Jumlah kalimat : {len(sentences)}")
    print(f"  Total karakter : {total_chars}")
    print(f"  Rata-rata      : {avg_chars:.1f} karakter/kalimat")
    print_memory_info("SETELAH LOAD DATA")

    # 2. Tokenizer (baru atau dari checkpoint)
    if resume_checkpoint:
        print("\n" + "="*60)
        print("📦 MEMUAT CHECKPOINT")
        print("="*60)
        model, optimizer, scheduler, tokenizer, config = load_checkpoint(resume_checkpoint)
        print(f"  Checkpoint     : {resume_checkpoint}")
        print(f"  Config         : {config}")
        print(f"  Vocab size     : {len(tokenizer.vocab)}")
        print(f"  Step terakhir  : {scheduler.step_num}")

        # Tambah langkah
        scheduler.total_steps = scheduler.step_num + ADDITIONAL_STEPS
        print(f"  Tambah steps   : {ADDITIONAL_STEPS}")
        print(f"  Total steps baru: {scheduler.total_steps}")
    else:
        # Training baru: latih tokenizer dari data
        print("\n" + "="*60)
        print("🔤 2. MELATIH TOKENIZER BPE")
        print("="*60)
        corpus = "\n".join(sentences)
        print(f"  Total karakter corpus : {len(corpus)}")
        print(f"  Target vocab size     : {VOCAB_SIZE}")
        t0 = time.time()
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(corpus, vocab_size=VOCAB_SIZE)
        print(f"  Tokenizer selesai     : {format_time(time.time()-t0)}")
        print(f"  Vocab size aktual     : {len(tokenizer.vocab)}")
        vocab_items = list(tokenizer.vocab.items())[:10]
        print(f"  10 token pertama      : {[t[0] for t in vocab_items]}")
        print_memory_info("SETELAH TOKENIZER")

    # 3. Encode & build dataset
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
        print("  ❌ Tidak ada contoh training! Periksa SEQ_LEN dan data.")
        return
    print_memory_info("SETELAH BUILD DATASET")

    # 4. Inisialisasi model (hanya jika training baru)
    if not resume_checkpoint:
        print("\n" + "="*60)
        print("🧠 4. INISIALISASI MODEL")
        print("="*60)
        actual_vocab = len(tokenizer.vocab)
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

        # Optimizer & scheduler baru
        print("\n" + "="*60)
        print("⚙️  5. OPTIMIZER & SCHEDULER")
        print("="*60)
        optimizer = AdamW(all_params, lr=LR, weight_decay=0.01)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS,
                                          total_steps=TOTAL_STEPS, base_lr=LR, min_lr=1e-5)
        # Tambahkan atribut total_steps agar konsisten
        scheduler.total_steps = TOTAL_STEPS
        print(f"  Optimizer      : AdamW")
        print(f"  LR awal        : {LR}")
        print(f"  Weight decay   : 0.01")
        print(f"  Scheduler      : Warmup + Cosine Decay")
        print(f"  Warmup steps   : {WARMUP_STEPS}")
        print(f"  Total steps    : {TOTAL_STEPS}")
        print(f"  Min LR         : 1e-5")
    else:
        # Resume: model, optimizer, scheduler sudah di-load
        print("\n" + "="*60)
        print("🔄 MELANJUTKAN TRAINING")
        print("="*60)
        model.set_training(True)
        print(f"  Model siap, training mode aktif.")
        print(f"  Optimizer & scheduler dari checkpoint.")

    # 6. TRAINING
    print("\n" + "="*60)
    print("🏋️  6. TRAINING")
    print("="*60)
    total_batches = max(1, len(examples) // BATCH_SIZE)
    print(f"  Dataset       : {len(examples)} contoh")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Batch/epoch   : {total_batches}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Max steps     : {scheduler.total_steps}")   # sekarang aman
    print(f"  Max grad norm : {MAX_GRAD_NORM}")

    start_time = time.time()
    global_step = scheduler.step_num
    best_loss = float('inf')
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

            if scheduler.step_num >= scheduler.total_steps:
                break

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, n_batches)
        print(f"  ✅ Epoch {epoch} selesai : "
              f"avg_loss={avg_loss:.4f}  best={best_loss:.4f}  "
              f"time={format_time(epoch_time)}  step={global_step}/{scheduler.total_steps}")
        print()

        history.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'lr': optimizer.lr,
            'step': global_step,
            'time': format_time(epoch_time)
        })

        if scheduler.step_num >= scheduler.total_steps:
            print(f"⏰ Training berhenti: mencapai batas {scheduler.total_steps} steps.")
            break

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 RINGKASAN TRAINING")
    print("="*60)
    print(f"  Total waktu      : {format_time(total_time)}")
    print(f"  Total steps      : {global_step}")
    print(f"  Best loss        : {best_loss:.4f}")
    print(f"  Last LR          : {optimizer.lr:.8f}")
    print(f"  Steps awal       : {scheduler.step_num - global_step + scheduler.step_num}")
    if history:
        print("\n  Riwayat per-epoch:")
        print("  Epoch  Avg Loss   Best Loss   LR           Waktu")
        print("  " + "-"*50)
        for h in history:
            print(f"  {h['epoch']:3d}    {h['avg_loss']:.4f}     {h['best_loss']:.4f}   {h['lr']:.8f}   {h['time']}")
    print_memory_info("SETELAH TRAINING")

    # 7. SAVE CHECKPOINT BARU
    print("\n" + "="*60)
    print("💾 7. MENYIMPAN CHECKPOINT BARU")
    print("="*60)
    version = get_next_version()
    checkpoint_path = f"Ai_{version}.json"
    config = {
        "vocab_size": len(tokenizer.vocab),
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "d_ff": model.d_ff,
        "max_len": model.max_len,
        "dropout": model.dropout
    }
    save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                    tokenizer=tokenizer, config=config)
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