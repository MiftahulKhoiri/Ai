#!/usr/bin/env python3
# demo.py - Script untuk chat dengan model MiniGPT yang sudah dilatih

import sys
import json
import os
import random
from typing import List, Tuple, Optional

from minigpt import (
    MiniGPT,
    ByteLevelBPETokenizer,
    generate,
    argmax,
    sample_top_k,
    sample_top_p
)

# ============================================================
# KONFIGURASI
# ============================================================
MAX_CONTEXT_TURNS = 5
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.2          # kecil untuk menghindari byte liar
TOP_P = 0.9
TOP_K = 20                 # batasi pilihan token
SHOW_REASONING = True

# ============================================================
# FUNGSI CHECKPOINT LOAD (copy dari training.py)
# ============================================================
def load_checkpoint(path: str):
    """Load checkpoint from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)
    
    # Restore tokenizer
    tokenizer = ByteLevelBPETokenizer()
    if 'vocab' in checkpoint and checkpoint['vocab']:
        vocab = {k: v for k, v in checkpoint['vocab'].items()}
        inv_vocab = {int(k): v for k, v in checkpoint['inv_vocab'].items()}
        merge_order = [tuple(pair) for pair in checkpoint.get('merge_order', [])]
        tokenizer.set_vocab(vocab)
        tokenizer.set_inv_vocab(inv_vocab)
        tokenizer.set_merge_order(merge_order)
    
    # Restore config
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', len(tokenizer.get_vocab()))
    d_model = config.get('d_model', 4)
    n_heads = config.get('n_heads', 2)
    n_layers = config.get('n_layers', 1)
    d_ff = config.get('d_ff', 16)
    max_len = config.get('max_len', 32)
    dropout = config.get('dropout', 0.1)
    
    # Create model
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    # Restore model parameters
    if 'params' in checkpoint and checkpoint['params']:
        params = model.parameters()
        for i, p in enumerate(params):
            if i < len(checkpoint['params']):
                p.data = checkpoint['params'][i]
    
    print(f"✅ Model loaded: vocab_size={vocab_size}, d_model={d_model}, layers={n_layers}")
    
    return model, None, None, tokenizer, config, None

# ============================================================
# FUNGSI UTILITY
# ============================================================
def sanitize_text(text: str) -> str:
    """Replace invalid characters with '�'"""
    return text.encode('utf-8', errors='replace').decode('utf-8')

def safe_generate(model, tokenizer, prompt: str, retries: int = 2) -> str:
    """Generate with fallback to greedy if error"""
    for attempt in range(retries + 1):
        try:
            temp = 0.0 if attempt == retries else TEMPERATURE
            
            # Pilih sampling method
            if temp > 0:
                # Gunakan sampling dengan top-k atau top-p
                # Library kita sudah punya generate dengan parameter default
                response = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=MAX_NEW_TOKENS
                )
            else:
                # Greedy (deterministik)
                response = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=MAX_NEW_TOKENS
                )
            
            # Pastikan response adalah string yang valid
            if isinstance(response, list):
                response = response[0] if response else ""
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

def format_prompt(history: List[str], user_input: str) -> str:
    """Format conversation history into prompt"""
    prompt_parts = []
    
    # Ambil context terakhir
    start_idx = max(0, len(history) - MAX_CONTEXT_TURNS * 2)
    for msg in history[start_idx:]:
        prompt_parts.append(msg)
    
    # Tambahkan input user
    prompt_parts.append(f"Pengguna: {user_input}\nAI:")
    return "\n".join(prompt_parts)

def clean_response(response: str) -> str:
    """Clean up generated response"""
    # Hapus bagian "AI:" jika ada
    if "AI:" in response:
        response = response.split("AI:")[-1].strip()
    
    # Hapus special tokens
    for cut in ["<bos>", "<eos>", "<pad>"]:
        if cut in response:
            response = response.split(cut)[0].strip()
    
    # Hapus bagian "Pengguna:" jika muncul
    if "Pengguna:" in response:
        response = response.split("Pengguna:")[0].strip()
    
    # Jika ada "Jawaban:" dan kita tidak menampilkan reasoning
    if not SHOW_REASONING and "Jawaban:" in response:
        final = response.split("Jawaban:")[-1].strip()
        # Ambil sampai titik atau newline
        for end_char in [".", "\n"]:
            if end_char in final:
                final = final.split(end_char)[0].strip()
                break
        response = final
    
    return response

# ============================================================
# FUNGSI CHAT
# ============================================================
def chat(model, tokenizer):
    """Main chat loop"""
    print("\n" + "="*60)
    print("🤖 MiniGPT Chatbot")
    print("="*60)
    print(f"  Vocab size    : {len(tokenizer.get_vocab())}")
    print(f"  Max context   : {MAX_CONTEXT_TURNS} turns")
    print(f"  Temperature   : {TEMPERATURE}")
    print(f"  Top-K         : {TOP_K}")
    print(f"  Top-P         : {TOP_P}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print("\nKetik 'exit' atau 'keluar' untuk berhenti.")
    print("-"*60)
    
    history = []
    total_tokens = 0
    
    while True:
        try:
            user_input = input("\nAnda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Sampai jumpa!")
            break
        
        # Cek exit
        if user_input.lower() in ("exit", "keluar", "quit", ""):
            print("👋 Sampai jumpa!")
            break
        
        # Format prompt
        prompt = format_prompt(history, user_input)
        
        # Generate response
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
        
        # Clean response
        response = clean_response(response)
        
        # Sanitize untuk keamanan output
        display_response = sanitize_text(response)
        if not display_response:
            display_response = "(Maaf, saya tidak bisa menjawab dengan baik. Coba pertanyaan lain.)"
        
        # Tampilkan response
        print(f"AI: {display_response}")
        
        # Update history
        history.append(f"Pengguna: {user_input}")
        history.append(f"AI: {response}")
        
        # Update token stats (approximate)
        total_tokens += len(prompt.split()) + len(response.split())
        
        # Limit history
        if len(history) > MAX_CONTEXT_TURNS * 4:
            history = history[-MAX_CONTEXT_TURNS * 4:]

# ============================================================
# FUNGSI TEST GENERATE
# ============================================================
def test_generate(model, tokenizer):
    """Test generation with sample prompts"""
    prompts = ["Halo", "Apa kabar", "Saya suka"]
    print("\n" + "="*60)
    print("🧪 TEST GENERATE")
    print("="*60)
    
    for i, p in enumerate(prompts, 1):
        try:
            print(f"\n[{i}] Prompt: {p}")
            result = generate(model, tokenizer, p, max_tokens=20)
            if isinstance(result, list):
                result = result[0] if result else ""
            print(f"    Hasil: {sanitize_text(result)}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    print()

# ============================================================
# MAIN
# ============================================================
def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Penggunaan: python3 demo.py <checkpoint.json>")
        print("Contoh: python3 demo.py Ai_1.json")
        return
    
    checkpoint_path = sys.argv[1]
    
    # Check file exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ File checkpoint '{checkpoint_path}' tidak ditemukan.")
        return
    
    # Load model
    print("="*60)
    print("📦 MEMUAT MODEL")
    print("="*60)
    
    try:
        model, _, _, tokenizer, config, _ = load_checkpoint(checkpoint_path)
        print(f"✅ Model berhasil dimuat dari: {checkpoint_path}")
        
        # Show config
        if config:
            print(f"   Config: {config}")
        
        # Test generate
        test_generate(model, tokenizer)
        
        # Start chat
        chat(model, tokenizer)
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: File checkpoint corrupt: {e}")
    except KeyError as e:
        print(f"❌ Error: Missing key in checkpoint: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()