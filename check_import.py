#!/usr/bin/env python3
"""Cek apa saja yang tersedia di modul minigpt"""

import minigpt

print("="*60)
print("📦 MODUL MINIGPT")
print("="*60)

# Daftar semua atribut
attrs = dir(minigpt)
print(f"\nTotal atribut: {len(attrs)}")
print("\nAtribut yang tersedia:")
for attr in sorted(attrs):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Cek class yang spesifik
print("\n" + "="*60)
print("🔍 CEK CLASS TERTENTU")
print("="*60)

classes_to_check = [
    'MiniGPT',
    'ByteLevelBPETokenizer', 
    'Tokenizer',
    'AdamW',
    'WarmupCosineScheduler',
    'Value'
]

for cls in classes_to_check:
    if hasattr(minigpt, cls):
        print(f"  ✅ {cls} tersedia")
    else:
        print(f"  ❌ {cls} TIDAK tersedia")

print("\n" + "="*60)