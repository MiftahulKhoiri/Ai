import sys
from minigpt import MiniGPT, ByteLevelBPETokenizer, load_checkpoint, generate

# ============================================================
# KONFIGURASI CHATBOT (dapat disesuaikan)
# ============================================================
MAX_CONTEXT_TURNS = 5      # jumlah pasangan tanya-jawab sebelumnya yang dijadikan konteks
MAX_NEW_TOKENS = 100        # maksimal token baru yang dihasilkan tiap respons
TEMPERATURE = 0.8           # kreativitas output (makin tinggi makin acak, 0 = deterministik)
TOP_P = 0.9                 # nucleus sampling: ambil token dengan prob kumulatif >= TOP_P
TOP_K = None                # top-k sampling (None = matikan, misal 40 untuk mengambil 40 token teratas)

# Template untuk membangun prompt gabungan dari seluruh percakapan
CHAT_TEMPLATE = "Pengguna: {user}\nAI: {ai}\n"


def chat(model, tokenizer):
    """Chatbot interaktif dengan memori percakapan terbatas."""
    print("\n=== MiniGPT Chatbot ===")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.\n")

    history = []  # daftar string, setiap elemen "Pengguna: ..." atau "AI: ..."

    while True:
        try:
            user_input = input("Anda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break

        if user_input.lower() in ("exit", "keluar", "quit", ""):
            print("Sampai jumpa!")
            break

        # Bangun prompt dari sebagian history terakhir + input baru
        prompt_parts = []
        start_idx = max(0, len(history) - MAX_CONTEXT_TURNS * 2)  # tiap giliran = 2 baris
        for msg in history[start_idx:]:
            prompt_parts.append(msg)
        prompt_parts.append(f"Pengguna: {user_input}\nAI:")

        prompt = "\n".join(prompt_parts)

        # Generate respons AI
        response = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )

        # Bersihkan respons: ambil hanya teks setelah "AI:" terakhir,
        # lalu potong jika muncul "Pengguna:" atau token spesial lagi.
        if "AI:" in response:
            response = response.split("AI:")[-1].strip()
        for cut in ["Pengguna:", "<bos>", "<eos>"]:
            if cut in response:
                response = response.split(cut)[0].strip()

        print(f"AI: {response}\n")

        # Simpan ke history
        history.append(f"Pengguna: {user_input}")
        history.append(f"AI: {response}")


def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python demo.py Ai_1.json")
        return

    checkpoint_path = sys.argv[1]
    tokenizer = ByteLevelBPETokenizer()
    # Buat model dummy, config sebenarnya akan dimuat dan menyesuaikan
    model = MiniGPT(vocab_size=100)
    config = load_checkpoint(checkpoint_path, model, tokenizer=tokenizer)
    print(f"Model {checkpoint_path} dimuat. Vocab size: {len(tokenizer.vocab)}")

    chat(model, tokenizer)


if __name__ == "__main__":
    main()