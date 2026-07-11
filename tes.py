 #!/usr/bin/env python3
# test_gradient.py - Test gradient flow

from minigpt import MiniGPT, Tokenizer, AdamW, cross_entropy_loss, clip_grad_norm

print("="*50)
print("🧪 TEST GRADIENT FLOW")
print("="*50)

# Buat model kecil
model = MiniGPT(vocab_size=100, d_model=4, n_heads=2, n_layers=1, d_ff=8, max_len=16)
model.set_training(True)

# Buat data
tokens = [1, 2, 3, 4, 5]
input_ids = tokens[:-1]
target_ids = tokens[1:]

print("1. Forward...")
logits = model.forward(input_ids)
print(f"   Logits: {len(logits)} positions")

print("\n2. Compute loss...")
loss = cross_entropy_loss(logits, target_ids, [])
print(f"   Loss: {loss.data}")

print("\n3. Backward...")
loss.backward()

print("\n4. Check gradients...")
params = model.parameters()
grad_norm = 0.0
has_grad = False
for i, p in enumerate(params[:10]):  # Cek 10 parameter pertama
    if p.grad != 0.0:
        has_grad = True
        print(f"   Param {i}: grad={p.grad:.6f}")
    grad_norm += p.grad * p.grad

if has_grad:
    print(f"\n✅ Gradien terdeteksi! (norm: {grad_norm**0.5:.6f})")
else:
    print("\n❌ TIDAK ADA gradien yang terdeteksi! Ada masalah di autograd.")