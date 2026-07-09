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
BATCH_SIZE = 4                   # jumlah contoh per batch (kecilkan jika OOM atau lambat)
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
    """Mengonversi detik ke string menit:detik."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def test_generate(model, tokenizer):
    """Tes singkat hasil generate setelah training."""
    prompts = ["Halo", "Apa kabar", "Saya suka"]
    print("\n--- Tes Generate ---")
    for p in prompts:
        result = generate(model, tokenizer, p, max_new_tokens=20)
        print(f"Prompt: {p!r} -> {result}")

def main():
    monitor_memory()
    print("Memuat data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    print(f"Jumlah kalimat: {len(sentences)}")

    # Gabungkan untuk training tokenizer
    corpus = "\n".join(sentences)
    print("Melatih tokenizer BPE...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(corpus, vocab_size=VOCAB_SIZE)
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Encode semua kalimat menjadi token ids dengan BOS/EOS
    tokenized = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_bos=True, add_eos=True)
        tokenized.append(ids)

    # Build dataset
    pad_id = tokenizer.vocab['<pad>']
    examples = build_dataset(tokenized, SEQ_LEN, pad_id)
    print(f"Jumlah contoh training: {len(examples)}")

    # Inisialisasi model
    model = MiniGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    model.set_training(True)  # <-- PERBAIKAN: gunakan set_training(True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS,
                                      total_steps=TOTAL_STEPS, base_lr=LR, min_lr=1e-5)

    print("Mulai training...")
    total_batches = max(1, len(examples) // BATCH_SIZE)
    print(f"Estimasi batch per epoch: {total_batches}")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{EPOCHS} dimulai ---")

        for batch in iter_batches(examples, BATCH_SIZE, shuffle=True):
            # Cek apakah batch kosong (seharusnya tidak terjadi)
            if len(batch) == 0:
                continue

            loss, grad_norm = train_batch(model, optimizer, batch, scheduler=scheduler,
                                          max_grad_norm=MAX_GRAD_NORM)
            total_loss += loss
            n_batches += 1

            # Tampilkan progress SETIAP BATCH agar terlihat pergerakan
            elapsed = time.time() - epoch_start
            est_remaining = max(0, total_batches - n_batches)
            eta_sec = (elapsed / n_batches) * est_remaining if n_batches > 0 else 0
            print(f"  Batch {n_batches}/{total_batches} | loss={loss:.4f} | grad={grad_norm:.4f} | "
                  f"lr={optimizer.lr:.6f} | {format_time(elapsed)} < {format_time(eta_sec)}")

            if scheduler.step_num >= TOTAL_STEPS:
                print("  Batas TOTAL_STEPS tercapai, menghentikan epoch lebih awal.")
                break

        avg_loss = total_loss / max(1, n_batches)
        epoch_time = time.time() - epoch_start
        print(f"  >>> Epoch {epoch}/{EPOCHS} selesai | avg loss: {avg_loss:.4f} | waktu: {format_time(epoch_time)}")

    total_time = time.time() - start_time
    print(f"\nTotal waktu training: {format_time(total_time)}")

    # Simpan checkpoint
    version = get_next_version()
    checkpoint_path = f"Ai_{version}.json"
    config = {
        "vocab_size": len(tokenizer.vocab),
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_len": MAX_LEN,
        "dropout": DROPOUT
    }
    save_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler,
                    tokenizer=tokenizer, config=config)
    print(f"Training selesai, model disimpan sebagai {checkpoint_path}")

    # Tes hasil training
    test_generate(model, tokenizer)

if __name__ == "__main__":
    main()