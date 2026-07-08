import json
from minigpt import MiniGPT, ByteLevelBPETokenizer, load_checkpoint, generate

# 1. Muat konfigurasi model dari checkpoint
with open("checkpoint_minigpt.json", "r") as f:
    ckpt = json.load(f)
cfg = ckpt["config"]

# 2. Inisialisasi tokenizer & model dengan arsitektur yang sama persis
tokenizer = ByteLevelBPETokenizer()
model = MiniGPT(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    max_len=cfg["max_len"],
    dropout=cfg["dropout"],
)

# 3. Muat bobot model + tokenizer dari checkpoint
load_checkpoint("checkpoint_minigpt.json", model, tokenizer=tokenizer)

print("✅ Model siap. Ketik 'exit' untuk keluar.\n")

# 4. Loop interaktif
while True:
    prompt = input("Prompt: ")
    if prompt.lower() == "exit":
        break
    # Hasilkan teks dengan sampling (pakai KV cache otomatis)
    hasil = generate(
        model, tokenizer, prompt,
        max_new_tokens=10,   # bisa disesuaikan
        temperature=0.9,
        top_p=0.9
    )
    print("Output:", hasil)