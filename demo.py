import sys
from minigpt import MiniGPT, ByteLevelBPETokenizer, generate
from minigpt_utils import load_checkpoint

# ============================================================
# KONFIGURASI (stabil untuk model yang sudah konvergen)
# ============================================================
MAX_CONTEXT_TURNS = 5
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.2          # tetap kecil untuk menghindari byte liar
TOP_P = 0.9
TOP_K = 20                 # batasi pilihan token
SHOW_REASONING = True

def sanitize_text(text):
    """Ganti semua karakter yang tidak dapat di-decode dengan '�'."""
    return text.encode('utf-8', errors='replace').decode('utf-8')

def safe_generate(model, tokenizer, prompt, retries=2):
    """Generate dengan fallback ke greedy jika error."""
    for attempt in range(retries + 1):
        try:
            temp = 0.0 if attempt == retries else TEMPERATURE
            response = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temp,
                top_k=TOP_K if temp > 0 else 0,
                top_p=TOP_P if temp > 0 else 1.0
            )
            # Pastikan response adalah string yang valid
            sanitize_text(response)  # cek apakah bisa di-encode
            return response
        except UnicodeDecodeError:
            if attempt == retries:
                raise
            continue
    return ""

def chat(model, tokenizer):
    print("\n=== MiniGPT Chatbot (Mode Stabil) ===")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.\n")
    history = []

    while True:
        try:
            user_input = input("Anda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break

        if user_input.lower() in ("exit", "keluar", "quit", ""):
            print("Sampai jumpa!")
            break

        prompt_parts = []
        start_idx = max(0, len(history) - MAX_CONTEXT_TURNS * 2)
        for msg in history[start_idx:]:
            prompt_parts.append(msg)
        prompt_parts.append(f"Pengguna: {user_input}\nAI:")
        prompt = "\n".join(prompt_parts)

        try:
            response = safe_generate(model, tokenizer, prompt)
        except UnicodeDecodeError:
            print("AI: Maaf, saya tidak dapat memproses permintaan ini. Coba lagi.\n")
            continue

        # Bersihkan respons
        if "AI:" in response:
            response = response.split("AI:")[-1].strip()
        for cut in ["<bos>", "<eos>"]:
            if cut in response:
                response = response.split(cut)[0].strip()
        if "Pengguna:" in response:
            response = response.split("Pengguna:")[0].strip()

        # Tampilkan dengan sanitasi
        if not SHOW_REASONING and "Jawaban:" in response:
            final = response.split("Jawaban:")[-1].strip()
            for end_char in [".", "\n"]:
                if end_char in final:
                    final = final.split(end_char)[0].strip()
            display = final
        else:
            display = response

        display = sanitize_text(display)  # <-- jaminan aman
        print(f"AI: {display}\n")

        history.append(f"Pengguna: {user_input}")
        history.append(f"AI: {response}")

def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python demo.py Ai_1.json")
        return

    checkpoint_path = sys.argv[1]
    model, _, _, tokenizer, config, _ = load_checkpoint(checkpoint_path)
    print(f"Model {checkpoint_path} dimuat. Vocab size: {len(tokenizer.vocab)}")
    chat(model, tokenizer)

if __name__ == "__main__":
    main()