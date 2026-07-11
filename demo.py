#!/usr/bin/env python3
# demo.py - Script untuk chat dengan model MiniGPT

import sys
import json
import os
import random
from typing import List, Tuple, Optional

from minigpt import (
    MiniGPT,
    Tokenizer,  # <-- PERUBAHAN: ByteLevelBPETokenizer → Tokenizer
    generate,
    AdvancedGenerator,
    GenerationConfig,
    argmax,
    sample_top_k,
    sample_top_p
)

# ============================================================
# KONFIGURASI
# ============================================================
MAX_CONTEXT_TURNS = 5
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 40
SHOW_REASONING = True

# ============================================================
# FUNGSI LOAD CHECKPOINT
# ============================================================
def load_checkpoint(path: str):
    """Load checkpoint from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    # Restore tokenizer
    tokenizer = Tokenizer()  # <-- PERUBAHAN
    if 'vocab' in checkpoint and checkpoint['vocab']:
        vocab = {k: v for k, v in checkpoint['vocab'].items()}
        inv_vocab = {int(k): v for k, v in checkpoint['inv_vocab'].items()}
        merge_order = [tuple(pair) for pair in checkpoint.get('merge_order', [])]
        tokenizer.set_vocab(vocab)
        tokenizer.set_inv_vocab(inv_vocab)
        tokenizer.set_merge_order(merge_order)
    
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', len(tokenizer.get_vocab()))
    d_model = config.get('d_model', 4)
    n_heads = config.get('n_heads', 2)
    n_layers = config.get('n_layers', 1)
    d_ff = config.get('d_ff', 16)
    max_len = config.get('max_len', 32)
    dropout = config.get('dropout', 0.1)
    
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    if 'params' in checkpoint and checkpoint['params']:
        params = model.parameters()
        for i, p in enumerate(params):
            if i < len(checkpoint['params']):
                p.data = checkpoint['params'][i]
    
    print(f"✅ Model loaded: vocab_size={vocab_size}, d_model={d_model}, layers={n_layers}")
    
    return model, None, None, tokenizer, config, None

# ============================================================
# GENERATE FUNCTION
# ============================================================
def generate_with_config(model, tokenizer, prompt: str, 
                         max_tokens: int = MAX_NEW_TOKENS,
                         temperature: float = TEMPERATURE,
                         top_k: int = TOP_K,
                         top_p: float = TOP_P) -> str:
    try:
        generator = AdvancedGenerator(model, tokenizer)
        config = GenerationConfig()
        config.max_length = max_tokens
        config.temperature = temperature
        config.top_k = top_k
        config.top_p = top_p
        config.num_beams = 1
        config.use_cache = True
        generator.set_config(config)
        results = generator.generate(prompt)
        if results and len(results) > 0:
            return results[0]
        return ""
    except Exception as e:
        print(f"⚠️  Advanced generate error: {e}, fallback ke basic generate")
        result = generate(model, tokenizer, prompt, max_tokens)
        if isinstance(result, list):
            return result[0] if result else ""
        return str(result)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def sanitize_text(text: str) -> str:
    return text.encode('utf-8', errors='replace').decode('utf-8')

def safe_generate(model, tokenizer, prompt: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            temp = 0.0 if attempt == retries else TEMPERATURE
            response = generate_with_config(
                model, tokenizer, prompt,
                max_tokens=MAX_NEW_TOKENS,
                temperature=temp,
                top_k=TOP_K if temp > 0 else 0,
                top_p=TOP_P if temp > 0 else 1.0
            )
            sanitize_text(response)
            return response
        except UnicodeDecodeError:
            if attempt == retries:
                raise
            continue
        except Exception as e:
            if attempt == retries:
                print(f"❌ Error generating: {e}")
                return ""
            continue
    return ""

# ============================================================
# CHAT FUNCTION
# ============================================================
def chat(model, tokenizer):
    print("\n" + "="*60)
    print("🤖 MiniGPT Chatbot")
    print("="*60)
    print(f"  Vocab size    : {len(tokenizer.get_vocab())}")
    print(f"  Temperature   : {TEMPERATURE}")
    print(f"  Top-K         : {TOP_K}")
    print(f"  Top-P         : {TOP_P}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print("\nKetik 'exit' atau 'keluar' untuk berhenti.")
    print("-"*60)
    
    history = []
    
    while True:
        try:
            user_input = input("\nAnda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Sampai jumpa!")
            break
        
        if user_input.lower() in ("exit", "keluar", "quit", ""):
            print("👋 Sampai jumpa!")
            break
        
        prompt_parts = []
        start_idx = max(0, len(history) - MAX_CONTEXT_TURNS * 2)
        for msg in history[start_idx:]:
            prompt_parts.append(msg)
        prompt_parts.append(f"Pengguna: {user_input}\nAI:")
        prompt = "\n".join(prompt_parts)
        
        try:
            print("🤔 Berpikir...", end="", flush=True)
            response = safe_generate(model, tokenizer, prompt)
            print("\r" + " " * 15 + "\r", end="", flush=True)
        except UnicodeDecodeError:
            print("\r❌ Error: Tidak dapat memproses permintaan. Coba lagi.")
            continue
        except Exception as e:
            print(f"\r❌ Error: {e}")
            continue
        
        if "AI:" in response:
            response = response.split("AI:")[-1].strip()
        for cut in ["<bos>", "<eos>"]:
            if cut in response:
                response = response.split(cut)[0].strip()
        if "Pengguna:" in response:
            response = response.split("Pengguna:")[0].strip()
        
        if not SHOW_REASONING and "Jawaban:" in response:
            final = response.split("Jawaban:")[-1].strip()
            for end_char in [".", "\n"]:
                if end_char in final:
                    final = final.split(end_char)[0].strip()
                    break
            response = final
        
        display = sanitize_text(response)
        if not display:
            display = "(Maaf, saya tidak bisa menjawab dengan baik. Coba pertanyaan lain.)"
        
        print(f"AI: {display}")
        history.append(f"Pengguna: {user_input}")
        history.append(f"AI: {response}")

# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python3 demo.py <checkpoint.json>")
        print("Contoh: python3 demo.py Ai_1.json")
        return
    
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File checkpoint '{checkpoint_path}' tidak ditemukan.")
        return
    
    print("="*60)
    print("📦 MEMUAT MODEL")
    print("="*60)
    
    try:
        model, _, _, tokenizer, config, _ = load_checkpoint(checkpoint_path)
        print(f"✅ Model berhasil dimuat dari: {checkpoint_path}")
        
        if config:
            print(f"   Config: {config}")
        
        # Test generate
        print("\n🧪 Test generate:")
        test_prompts = ["Halo", "Apa kabar"]
        for p in test_prompts:
            try:
                result = generate(model, tokenizer, p, max_tokens=20)
                if isinstance(result, list):
                    result = result[0] if result else ""
                print(f"  Prompt: {p} → {sanitize_text(result)}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        chat(model, tokenizer)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()