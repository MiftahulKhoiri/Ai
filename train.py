"""
Train Mini-GPT menggunakan library minigpt.py.
Script ini melatih model dari nol, menyimpan checkpoint, lalu mencoba generate teks.
Pastikan file minigpt.py berada di direktori yang sama.
"""

import math
import random
import json
import gc
from minigpt import (
    ByteLevelBPETokenizer,
    MiniGPT,
    AdamW,
    WarmupCosineScheduler,
    build_dataset,
    iter_batches,
    train_batch,
    save_checkpoint,
    load_checkpoint,
    generate,
)

# ============================================================
# KONFIGURASI
# ============================================================
SEED = 0
EPOCHS = 6
BATCH_SIZE = 2
SEQ_LEN = 10
VOCAB_SIZE = 320
D_MODEL = 12
N_HEADS = 2
N_LAYERS = 1
D_FF = 24
MAX_LEN = 32
DROPOUT = 0.1
LR = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
CHECKPOINT_PATH = "checkpoint_minigpt.json"

# Data contoh (bisa diganti dengan file teks sendiri)
KALIMAT = [
    "kucing suka makan ikan.",
    "anjing suka makan tulang.",
    "kucing dan anjing adalah hewan peliharaan.",
    "burung suka terbang tinggi di langit.",
    "ikan berenang di air.",
    "kucing suka tidur di sofa.",
]


def main():
    random.seed(SEED)

    # 1. Tokenizer
    print("=== Melatih Tokenizer Byte-Level BPE ===")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(" ".join(KALIMAT), vocab_size=VOCAB_SIZE)
    print(f"Ukuran vocab: {len(tokenizer.vocab)}")

    # Uji tokenizer
    enc = tokenizer.encode("kucing suka makan ikan 🐟")
    print(f"Encode (dgn emoji) -> {enc}")
    print(f"Decode kembali     -> '{tokenizer.decode(enc)}'")

    # 2. Dataset
    print("\n=== Menyiapkan Dataset (BOS/EOS + input/target eksplisit) ===")
    pad_id = tokenizer.vocab['<pad>']
    token_lists = [tokenizer.encode(k, add_bos=True, add_eos=True) for k in KALIMAT]
    dataset = build_dataset(token_lists, seq_len=SEQ_LEN, pad_id=pad_id)
    print(f"Jumlah contoh training: {len(dataset)}")

    # 3. Model
    print("\n=== Membangun Mini-GPT ===")
    model = MiniGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    )
    model.train()
    params = model.parameters()
    print(f"Jumlah parameter (Value): {len(params)}")

    # 4. Optimizer & Scheduler
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    n_batches_per_epoch = math.ceil(len(dataset) / BATCH_SIZE)
    total_steps = EPOCHS * n_batches_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=max(2, total_steps // 5),
        total_steps=total_steps,
        base_lr=LR,
        min_lr=0.002,
    )

    # 5. Training Loop
    print("\n=== Training (batch + AdamW + gradient clipping + scheduler) ===")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batch = 0
        for batch in iter_batches(dataset, batch_size=BATCH_SIZE, shuffle=True):
            loss_val, grad_norm = train_batch(
                model, optimizer, batch,
                scheduler=scheduler,
                max_grad_norm=MAX_GRAD_NORM,
            )
            epoch_loss += loss_val
            n_batch += 1

        # Logging setiap beberapa epoch
        if epoch == 1 or epoch % 3 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:2d}/{EPOCHS} - "
                f"Loss: {epoch_loss / n_batch:.4f} - "
                f"lr: {optimizer.lr:.5f} - "
                f"grad_norm terakhir: {grad_norm:.3f}"
            )
        gc.collect()  # lepaskan graph epoch sebelumnya

    # 6. Simpan Checkpoint
    print("\n=== Menyimpan Checkpoint ===")
    model_config = {
        'vocab_size': len(tokenizer.vocab),
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'd_ff': D_FF,
        'max_len': MAX_LEN,
        'dropout': DROPOUT,
    }
    save_checkpoint(
        CHECKPOINT_PATH,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        config=model_config,
    )

    # 7. Muat Kembali & Verifikasi
    print("\n=== Memuat Ulang Checkpoint ke Model BARU (uji save/load) ===")
    with open(CHECKPOINT_PATH, 'r') as f:
        cfg = json.load(f)['config']
    model2 = MiniGPT(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        max_len=cfg['max_len'],
        dropout=cfg['dropout'],
    )
    tokenizer2 = ByteLevelBPETokenizer()
    load_checkpoint(CHECKPOINT_PATH, model2, tokenizer=tokenizer2)

    # 8. Generate Demo
    print("\n=== Generate Teks dengan Model Hasil Load (KV cache aktif) ===")
    prompt = "kucing suka"
    hasil = generate(
        model2, tokenizer2, prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9,
    )
    print(f"Prompt : {prompt}")
    print(f"Output : {hasil}")


if __name__ == "__main__":
    main()