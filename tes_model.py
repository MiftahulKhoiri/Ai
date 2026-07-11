#!/usr/bin/env python3
# test_value.py - Test Value autograd

from minigpt import Value

print("="*50)
print("🧪 TEST VALUE AUTOGRAD")
print("="*50)

# Test 1: Penjumlahan
print("\n1. Test penjumlahan:")
a = Value(5.0)
b = Value(3.0)
c = a + b
print(f"   a={a.data}, b={b.data}, c={c.data}")
c.backward()
print(f"   a.grad={a.grad}, b.grad={b.grad}")
print(f"   ✅ a.grad=1.0, b.grad=1.0" if abs(a.grad-1.0)<0.001 and abs(b.grad-1.0)<0.001 else "   ❌ GAGAL!")

# Test 2: Perkalian
print("\n2. Test perkalian:")
a = Value(5.0)
b = Value(3.0)
c = a * b
print(f"   a={a.data}, b={b.data}, c={c.data}")
c.backward()
print(f"   a.grad={a.grad}, b.grad={b.grad}")
print(f"   ✅ a.grad=3.0, b.grad=5.0" if abs(a.grad-3.0)<0.001 and abs(b.grad-5.0)<0.001 else "   ❌ GAGAL!")

# Test 3: Kombinasi
print("\n3. Test kombinasi:")
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)
d = a * b + c
print(f"   a={a.data}, b={b.data}, c={c.data}, d={d.data}")
d.backward()
print(f"   a.grad={a.grad}, b.grad={b.grad}, c.grad={c.grad}")
print(f"   ✅ a.grad=3.0, b.grad=2.0, c.grad=1.0" if abs(a.grad-3.0)<0.001 and abs(b.grad-2.0)<0.001 and abs(c.grad-1.0)<0.001 else "   ❌ GAGAL!")

print("\n" + "="*50)