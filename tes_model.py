#!/usr/bin/env python3
# test_model.py - Test model tanpa training

import sys
from minigpt import MiniGPT, Tokenizer, cross_entropy_loss, Value

print("="*60)
print("🧪 TEST MODEL SEDERHANA")
print("="*60)

# 1. Buat tokenizer
print("\n1. Membuat tokenizer...")
tokenizer = Tokenizer()
tokenizer.train("Halo dunia ini adalah test", 50)
print(f"   Vocab size: {tokenizer.vocab_size()}")

# 2. Buat model kecil
print("\n2. Membuat model...")
model = MiniGPT(
    vocab_size=tokenizer.vocab_size(),
    d_model=4,
    n_heads=2,
    n_layers=1,
    d_ff=8,
    max_len=16,
    dropout=0.1
)
print(f"   Model created: d_model={model.d_model}, max_len={model.max_len}, vocab_size={model.vocab_size}")

# 3. Test forward
print("\n3. Test forward...")
text = "Halo dunia"
tokens = tokenizer.encode(text)
print(f"   Tokens: {tokens}")

try:
    logits = model.forward(tokens)
    print(f"   ✅ Forward success! Logits: {len(logits)} positions")
    if len(logits) > 0:
        print(f"   Logits[0] shape: {len(logits[0])} tokens")
except Exception as e:
    print(f"   ❌ Forward error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test backward
print("\n4. Test backward...")
try:
    if len(tokens) > 1:
        target = tokens[1:] + [0]  # target shift
        print(f"   Target: {target}")
        loss = cross_entropy_loss(logits, target, [])
        print(f"   Loss: {loss.data}")
        
        loss.backward()
        print(f"   ✅ Backward success!")
    else:
        print("   ⚠️ Tokens terlalu pendek untuk backward test")
except Exception as e:
    print(f"   ❌ Backward error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Semua test berhasil!")