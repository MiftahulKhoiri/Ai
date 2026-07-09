import json
import os
import glob
from minigpt import (
    ByteLevelBPETokenizer, MiniGPT, AdamW, WarmupCosineScheduler,
    build_dataset, iter_batches, train_batch, save_checkpoint, generate
)
from demo import test_generate  # untuk tes hasil training

# ============================================================
# KONFIGURASI TRAINING (dapat disesuaikan)
# ============================================================
DATA_FILE = "data.json"          # file JSON berisi list kalimat latih (string)
SEQ_LEN = 32                      # panjang maksimal urutan token dalam satu contoh (sliding window)
BATCH_SIZE = 16                   # jumlah contoh per batch training
EPOCHS = 10                       # berapa kali seluruh data dilewati (epoch)
LR = 0.01                         # learning rate awal (sebelum warmup/cosine decay)
WARMUP_STEPS = 50                 # jumlah langkah pemanasan (linear naik dari 0 ke LR)
TOTAL_STEPS = 500                 # total langkah training (setelah ini scheduler cosine turun ke min_lr)
MAX_GRAD_NORM = 1.0               # batas maksimal norm gradien (gradient clipping)
D_MODEL = 16                      # dimensi embedding dan hidden state transformer
N_HEADS = 2                       # jumlah head dalam multi-head attention
N_LAYERS = 2                      # jumlah layer transformer (blok)
D_FF = 32                         # dimensi feed-forward inner layer
MAX_LEN = 64                      # panjang maksimum posisi (max position embedding)
DROPOUT = 0.1                     # probabilitas dropout (regularisasi)
VOCAB_SIZE = 400                  # ukuran kosakata tokenizer BPE (termasuk special tokens)

def get_next_version():
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

def main():
    print("Memuat data...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        sentences = json.load(f)  # list of strings
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
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS,
                                      total_steps=TOTAL_STEPS, base_lr=LR, min_lr=1e-5)

    print("Mulai training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches = 0
        for batch in iter_batches(examples, BATCH_SIZE, shuffle=True):
            loss, grad_norm = train_batch(model, optimizer, batch, scheduler=scheduler,
                                          max_grad_norm=MAX_GRAD_NORM)
            total_loss += loss
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"Epoch {epoch+1}, batch {n_batches}: loss={loss:.4f}, grad_norm={grad_norm:.4f}, lr={optimizer.lr:.6f}")
            if scheduler.step_num >= TOTAL_STEPS:
                break
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1} selesai, avg loss: {avg_loss:.4f}")

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

    # Tes hasil training menggunakan demo
    print("\n--- Tes generate ---")
    test_generate(model, tokenizer)

if __name__ == "__main__":
    main()