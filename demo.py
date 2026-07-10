import sys
from minigpt import MiniGPT, ByteLevelBPETokenizer, generate
from minigpt_utils import load_checkpoint

# ============================================================
# KONFIGURASI CHATBOT (dapat disesuaikan)
# ============================================================
MAX_CONTEXT_TURNS = 5      # jumlah pasangan tanya-jawab sebelumnya yang dijadikan konteks
MAX_NEW_TOKENS = 100       # maksimal token baru yang dihasilkan tiap respons
TEMPERATURE = 0.8          # kreativitas output (makin tinggi makin acak, 0 = deterministik)
TOP_P = 0.9                # nucleus sampling: ambil token dengan prob kumulatif >= TOP_P
TOP_K = 0                  # top-k sampling (0 = matikan, misal 40 untuk mengambil 40 token teratas)
SHOW_REASONING = True      # tampilkan seluruh langkah berpikir? False = hanya jawaban akhir

def chat(model, tokenizer):
    """Chatbot interaktif dengan memori percakapan terbatas."""
    print("\n=== MiniGPT Chatbot (Reasoning Mode) ===")
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

        # Bangun prompt dari history + input baru
        prompt_parts = []
        start_idx = max(0, len(history) - MAX_CONTEXT_TURNS * 2)
        for msg in history[start_idx:]:
            prompt_parts.append(msg)
        prompt_parts.append(f"Pengguna: {user_input}\nAI:")

        prompt = "\n".join(prompt_parts)

        # Generate respons AI (model akan melanjutkan dari "AI:")
        response = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )

        # Ambil bagian setelah "AI:" terakhir (dari prompt gabungan + generasi)
        if "AI:" in response:
            response = response.split("AI:")[-1].strip()

        # Potong hanya jika ada token khusus atau muncul "Pengguna:" baru
        for cut in ["<bos>", "<eos>"]:
            if cut in response:
                response = response.split(cut)[0].strip()
        # Potong jika ada "Pengguna:" (mencegah model berhalusinasi giliran berikutnya)
        if "Pengguna:" in response:
            response = response.split("Pengguna:")[0].strip()

        # Opsi: hanya tampilkan jawaban akhir
        if not SHOW_REASONING and "Jawaban:" in response:
            final = response.split("Jawaban:")[-1].strip()
            for end_char in [".", "\n"]:
                if end_char in final:
                    final = final.split(end_char)[0].strip()
            display = final
        else:
            display = response

        print(f"AI: {display}\n")

        # Simpan ke history (simpan versi lengkapnya)
        history.append(f"Pengguna: {user_input}")
        history.append(f"AI: {response}")


def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python demo.py Ai_1.json")
        return

    checkpoint_path = sys.argv[1]

    # load_checkpoint mengembalikan 6 nilai, kita hanya butuh model, tokenizer, config
    model, _, _, tokenizer, config, _ = load_checkpoint(checkpoint_path)

    print(f"Model {checkpoint_path} dimuat. Vocab size: {len(tokenizer.vocab)}")

    chat(model, tokenizer)


if __name__ == "__main__":
    main()