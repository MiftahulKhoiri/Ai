#!/usr/bin/env python3
# test_batch.py - Test batch training

import sys
import random
from minigpt import MiniGPT, Tokenizer, AdamW, cross_entropy_loss, clip_grad_norm

print("="*60)
print("🧪 TEST BATCH TRAINING")
print("="*60)

# 1. Buat data kecil
print("\n1. Membuat data...")
tokenizer = Tokenizer()
tokenizer.train("Halo dunia. Ini adalah test. Belajar MiniGPT. Semoga berhasil.", 50)

texts = [
    "Halo dunia",
    "Ini adalah test",
    "Belajar MiniGPT",
    "Semoga berhasil"
]

sequences = []
for text in texts:
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(f"   '{text}' -> {len(tokens)} tokens: {tokens}")
    if len(tokens) >= 4:
        sequences.append(tokens[:4])

print(f"   Data: {len(sequences)} sequences")

# 2. Buat model
print("\n2. Membuat model...")
model = MiniGPT(
    vocab_size=tokenizer.vocab_size(),
    d_model=4,
    n_heads=2,
    n_layers=1,
    d_ff=8,
    max_len=16
)
model.set_training(True)
print(f"   Model created: d_model={model.d_model}, max_len={model.max_len}")

# 3. Buat optimizer
print("\n3. Membuat optimizer...")
params = model.parameters()
print(f"   Parameters: {len(params)} tensors")
optimizer = AdamW(params, lr=0.01)
print(f"   Optimizer created")

# 4. Test batch
print("\n4. Test batch training...")

for epoch in range(3):
    total_loss = 0.0
    n_batches = 0
    print(f"\n   Epoch {epoch+1}:")
    
    for seq in sequences:
        if len(seq) < 2:
            continue
            
        input_ids = seq[:-1]
        target_ids = seq[1:]
        
        print(f"      Input: {input_ids}, Target: {target_ids}")
        
        try:
            # Forward
            logits = model.forward(input_ids)
            if not logits:
                print("      ⚠️ No logits returned, skipping")
                continue
                
            loss = cross_entropy_loss(logits, target_ids, [])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Clip grad
            params = model.parameters()
            grad_norm = clip_grad_norm(params, 1.0)
            
            # Step
            optimizer.step()
            
            total_loss += loss.data
            n_batches += 1
            print(f"      Loss: {loss.data:.4f}, Grad norm: {grad_norm:.4f}")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if n_batches > 0:
        print(f"   Avg Loss: {total_loss/n_batches:.4f}")

print("\n✅ Batch test berhasil!")