#!/usr/bin/env python3
"""
generate_data.py - Generator data training untuk MiniGPT
Menghasilkan data dengan berbagai jenis reasoning dan dialog
"""

import json
import random
import sys
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime

# ============================================================
# KONFIGURASI
# ============================================================
CONFIG = {
    'total_data': 2000,           # Total data yang dihasilkan
    'output_file': 'data.json',    # Nama file output
    'seed': 42,                    # Random seed untuk reproducibility
    'min_langkah': 2,              # Minimum langkah reasoning
    'max_langkah': 5,              # Maximum langkah reasoning
}

# ============================================================
# KATEGORI DATA
# ============================================================
KATEGORI = {
    'aritmatika': 25,      # 25%
    'pengetahuan': 20,     # 20%
    'logika': 15,          # 15%
    'prosedur': 15,        # 15%
    'dialog': 10,          # 10%
    'cerita': 10,          # 10%
    'definisi': 5,         # 5%
}

# ============================================================
# DATA POOLS
# ============================================================
NAMA = ["Andi", "Budi", "Cici", "Dedi", "Eka", "Fani", "Gita", "Hadi", "Indah", "Joko"]
TEMPAT = ["pasar", "sekolah", "taman", "rumah sakit", "perpustakaan", "stasiun", "bandara", "restoran"]
ALAT = ["motor", "mobil", "bus", "kereta", "pesawat", "kapal", "sepeda", "jalan kaki"]
HEWAN = ["kucing", "anjing", "sapi", "domba", "ayam", "bebek", "kambing", "kelinci"]
BUAH = ["apel", "jeruk", "pisang", "mangga", "anggur", "semangka", "pepaya", "nanas"]
WARNA = ["merah", "biru", "hijau", "kuning", "ungu", "oranye", "hitam", "putih"]
MAKANAN = ["nasi", "mie", "roti", "kue", "bakso", "soto", "gado-gado", "nasi goreng"]

# ============================================================
# FUNGSI PEMBANTU
# ============================================================
def random_choice(items):
    """Pilih random dari list"""
    return items[random.randint(0, len(items) - 1)]

def random_int(min_val, max_val):
    """Random integer antara min dan max"""
    return random.randint(min_val, max_val)

def format_list(items, conjunction="dan"):
    """Format list menjadi string: 'a, b, dan c'"""
    if len(items) == 0:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"

# ============================================================
# GENERATOR DATA PER KATEGORI
# ============================================================

def generate_aritmatika(langkah_count=None) -> str:
    """Data aritmatika dengan multiple steps"""
    if langkah_count is None:
        langkah_count = random_int(CONFIG['min_langkah'], min(3, CONFIG['max_langkah']))
    
    # Pilih operasi
    op = random.choice(['+', '-', '*'])
    numbers = [random_int(1, 20) for _ in range(langkah_count)]
    
    # Pastikan hasil positif untuk pengurangan
    if op == '-':
        numbers.sort(reverse=True)
    
    # Buat langkah-langkah
    langkahs = []
    current = numbers[0]
    langkahs.append(f"Langkah 1: Mulai dengan angka {current}")
    
    for i in range(1, len(numbers)):
        if op == '+':
            next_val = current + numbers[i]
            langkahs.append(f"Langkah {i+1}: {current} + {numbers[i]} = {next_val}")
            current = next_val
        elif op == '-':
            next_val = current - numbers[i]
            langkahs.append(f"Langkah {i+1}: {current} - {numbers[i]} = {next_val}")
            current = next_val
        else:  # *
            next_val = current * numbers[i]
            langkahs.append(f"Langkah {i+1}: {current} × {numbers[i]} = {next_val}")
            current = next_val
    
    # Format pertanyaan
    if op == '+':
        soal = f"Berapa {numbers[0]} + " + " + ".join(str(n) for n in numbers[1:]) + "?"
    elif op == '-':
        soal = f"Berapa {numbers[0]} - " + " - ".join(str(n) for n in numbers[1:]) + "?"
    else:
        soal = f"Berapa {numbers[0]} × " + " × ".join(str(n) for n in numbers[1:]) + "?"
    
    langkahs.append(f"Jawaban: {current}")
    
    return f"Pengguna: {soal}\nAI: " + " ".join(langkahs) + "\n"


def generate_pengetahuan(langkah_count=None) -> str:
    """Data pengetahuan dengan reasoning"""
    templates = [
        # Fakta tentang hewan
        lambda: (
            f"Pengguna: Apa yang kamu ketahui tentang {random_choice(HEWAN)}?\n"
            f"AI: Langkah 1: {random_choice(HEWAN)} adalah hewan yang hidup di darat. "
            f"Langkah 2: {random_choice(HEWAN)} memiliki {random_choice(['bulu', 'rambut', 'kulit'])}. "
            f"Langkah 3: {random_choice(HEWAN)} biasanya memakan {random_choice(['rumput', 'daging', 'bijian'])}. "
            f"Jawaban: {random_choice(HEWAN)} adalah hewan darat yang memiliki {random_choice(['bulu', 'rambut', 'kulit'])}.\n"
        ),
        # Fakta tentang buah
        lambda: (
            f"Pengguna: Informasi tentang {random_choice(BUAH)}.\n"
            f"AI: Langkah 1: {random_choice(BUAH)} adalah buah yang berwarna {random_choice(WARNA)}. "
            f"Langkah 2: {random_choice(BUAH)} memiliki rasa {random_choice(['manis', 'asam', 'segar'])}. "
            f"Langkah 3: {random_choice(BUAH)} mengandung vitamin {random_choice(['A', 'C', 'B'])}. "
            f"Jawaban: {random_choice(BUAH)} adalah buah {random_choice(WARNA)} yang mengandung vitamin {random_choice(['A', 'C', 'B'])}.\n"
        ),
        # Fakta tentang tempat
        lambda: (
            f"Pengguna: Ceritakan tentang {random_choice(TEMPAT)}.\n"
            f"AI: Langkah 1: {random_choice(TEMPAT)} adalah tempat yang sering dikunjungi orang. "
            f"Langkah 2: Di {random_choice(TEMPAT)}, orang bisa {random_choice(['belanja', 'belajar', 'bermain', 'bekerja'])}. "
            f"Langkah 3: {random_choice(TEMPAT)} biasanya buka dari pagi sampai sore. "
            f"Jawaban: {random_choice(TEMPAT)} adalah tempat untuk {random_choice(['belanja', 'belajar', 'bermain', 'bekerja'])}.\n"
        ),
    ]
    return random.choice(templates)()


def generate_logika(langkah_count=None) -> str:
    """Data logika dan penalaran"""
    templates = [
        # Perbandingan
        lambda: (
            f"Pengguna: Mana yang lebih besar, {random_choice(BUAH)} atau {random_choice(BUAH)}?\n"
            f"AI: Langkah 1: Membandingkan dua buah: {random_choice(BUAH)} dan {random_choice(BUAH)}. "
            f"Langkah 2: Ukuran {random_choice(BUAH)} lebih besar dari {random_choice(BUAH)}. "
            f"Jawaban: {random_choice(BUAH)} lebih besar.\n"
        ),
        # Urutan
        lambda: (
            f"Pengguna: Urutkan dari terkecil: {random_choice(BUAH)}, {random_choice(BUAH)}, {random_choice(BUAH)}.\n"
            f"AI: Langkah 1: Mengidentifikasi ukuran masing-masing buah. "
            f"Langkah 2: Mengurutkan dari yang terkecil. "
            f"Jawaban: {random_choice(BUAH)}, {random_choice(BUAH)}, {random_choice(BUAH)}.\n"
        ),
        # Jika-maka
        lambda: (
            f"Pengguna: Jika hari hujan, apa yang harus dilakukan?\n"
            f"AI: Langkah 1: Identifikasi situasi: hari hujan. "
            f"Langkah 2: Cari solusi untuk menghadapi hujan. "
            f"Langkah 3: Siapkan payung atau jas hujan. "
            f"Jawaban: Bawa payung atau jas hujan.\n"
        ),
    ]
    return random.choice(templates)()


def generate_prosedur(langkah_count=None) -> str:
    """Data prosedur/langkah-langkah"""
    if langkah_count is None:
        langkah_count = random_int(3, min(5, CONFIG['max_langkah']))
    
    templates = [
        # Cara membuat
        lambda: (
            f"Pengguna: Bagaimana cara membuat {random_choice(MAKANAN)}?\n"
            f"AI: Langkah 1: Siapkan bahan-bahan untuk {random_choice(MAKANAN)}. "
            f"Langkah 2: Potong dan cuci semua bahan. "
            f"Langkah 3: Masak dengan api sedang. "
            f"Langkah 4: Tambahkan bumbu sesuai selera. "
            f"Langkah 5: Sajikan dengan {random_choice(['nasi', 'roti', 'kentang'])}. "
            f"Jawaban: {random_choice(MAKANAN)} siap disajikan.\n"
        ),
        # Cara pergi
        lambda: (
            f"Pengguna: Bagaimana cara pergi ke {random_choice(TEMPAT)}?\n"
            f"AI: Langkah 1: Tentukan tujuan: {random_choice(TEMPAT)}. "
            f"Langkah 2: Pilih kendaraan: {random_choice(ALAT)}. "
            f"Langkah 3: Ikuti petunjuk arah. "
            f"Langkah 4: Tiba di {random_choice(TEMPAT)}. "
            f"Jawaban: Naik {random_choice(ALAT)} ke {random_choice(TEMPAT)}.\n"
        ),
    ]
    return random.choice(templates)()


def generate_dialog(langkah_count=None) -> str:
    """Data dialog percakapan"""
    templates = [
        # Sapaan
        lambda: (
            f"Pengguna: Halo, apa kabar?\n"
            f"AI: Langkah 1: Mendengar sapaan dari pengguna. "
            f"Langkah 2: Memberikan respons yang ramah. "
            f"Jawaban: Halo! Saya baik, terima kasih. Bagaimana dengan Anda?\n"
        ),
        # Perkenalan
        lambda: (
            f"Pengguna: Siapa nama Anda?\n"
            f"AI: Langkah 1: Memahami pertanyaan tentang identitas. "
            f"Langkah 2: Memperkenalkan diri. "
            f"Jawaban: Nama saya MiniGPT, asisten AI yang siap membantu.\n"
        ),
        # Bantuan
        lambda: (
            f"Pengguna: Bisakah Anda membantu saya?\n"
            f"AI: Langkah 1: Memahami permintaan bantuan. "
            f"Langkah 2: Menawarkan bantuan. "
            f"Jawaban: Tentu! Saya siap membantu Anda. Ada yang bisa saya bantu?\n"
        ),
    ]
    return random.choice(templates)()


def generate_cerita(langkah_count=None) -> str:
    """Data cerita pendek"""
    if langkah_count is None:
        langkah_count = random_int(3, min(4, CONFIG['max_langkah']))
    
    nama = random_choice(NAMA)
    tempat = random_choice(TEMPAT)
    alat = random_choice(ALAT)
    
    cerita = f"Pengguna: Ceritakan tentang {nama} pergi ke {tempat}.\n"
    cerita += "AI: "
    
    langkahs = [
        f"Langkah 1: {nama} bangun pagi dan bersiap-siap.",
        f"Langkah 2: {nama} naik {alat} menuju {tempat}.",
        f"Langkah 3: {nama} tiba di {tempat} dan melakukan aktivitas.",
        f"Langkah 4: {nama} pulang dengan selamat.",
    ]
    
    # Ambil langkah sesuai jumlah
    for i in range(min(langkah_count, len(langkahs))):
        cerita += langkahs[i] + " "
    
    cerita += f"Jawaban: {nama} pergi ke {tempat} dan kembali.\n"
    return cerita


def generate_definisi(langkah_count=None) -> str:
    """Data definisi kata"""
    templates = [
        lambda: (
            f"Pengguna: Apa itu {random_choice(HEWAN)}?\n"
            f"AI: Langkah 1: {random_choice(HEWAN)} adalah jenis hewan. "
            f"Langkah 2: {random_choice(HEWAN)} memiliki ciri khas {random_choice(['bulu', 'rambut', 'kulit'])}. "
            f"Langkah 3: {random_choice(HEWAN)} hidup di {random_choice(['darat', 'air', 'udara'])}. "
            f"Jawaban: {random_choice(HEWAN)} adalah hewan yang hidup di {random_choice(['darat', 'air', 'udara'])}.\n"
        ),
        lambda: (
            f"Pengguna: Jelaskan tentang {random_choice(BUAH)}.\n"
            f"AI: Langkah 1: {random_choice(BUAH)} adalah jenis buah. "
            f"Langkah 2: {random_choice(BUAH)} berwarna {random_choice(WARNA)}. "
            f"Langkah 3: {random_choice(BUAH)} mengandung vitamin. "
            f"Jawaban: {random_choice(BUAH)} adalah buah berwarna {random_choice(WARNA)}.\n"
        ),
    ]
    return random.choice(templates)()


# ============================================================
# GENERATOR UTAMA
# ============================================================
def generate_data(total: int, seed: int = None) -> List[str]:
    """Generate data training dengan berbagai kategori"""
    if seed is not None:
        random.seed(seed)
    
    data = []
    
    # Hitung jumlah per kategori
    kategori_percent = list(KATEGORI.values())
    total_percent = sum(kategori_percent)
    
    # Normalisasi
    target_per_kategori = {}
    for name, percent in KATEGORI.items():
        target_per_kategori[name] = int(total * percent / total_percent)
    
    # Tambahkan sisa ke kategori pertama
    remaining = total - sum(target_per_kategori.values())
    if remaining > 0:
        first_category = list(KATEGORI.keys())[0]
        target_per_kategori[first_category] += remaining
    
    # Generate data per kategori
    generators = {
        'aritmatika': generate_aritmatika,
        'pengetahuan': generate_pengetahuan,
        'logika': generate_logika,
        'prosedur': generate_prosedur,
        'dialog': generate_dialog,
        'cerita': generate_cerita,
        'definisi': generate_definisi,
    }
    
    for kategori, count in target_per_kategori.items():
        for _ in range(count):
            if kategori in generators:
                # Random langkah_count untuk variasi
                langkah_count = random_int(CONFIG['min_langkah'], CONFIG['max_langkah'])
                data.append(generators[kategori](langkah_count))
    
    # Shuffle data
    random.shuffle(data)
    
    return data


def generate_mixed_data(total: int = 2000) -> List[str]:
    """Generate data dengan campuran berbagai jenis"""
    data = []
    
    # Data reasoning (40%)
    reasoning_count = int(total * 0.4)
    data.extend(generate_data(reasoning_count))
    
    # Data tanya jawab sederhana (30%)
    qa_count = int(total * 0.3)
    for _ in range(qa_count):
        pertanyaan = random.choice([
            f"Apa warna {random_choice(BUAH)}?",
            f"Di mana {random_choice(NAMA)} tinggal?",
            f"Kapan waktu yang tepat untuk {random_choice(['belajar', 'bermain', 'bekerja'])}?",
        ])
        jawaban = random.choice([
            f"{random_choice(WARNA)}",
            f"di {random_choice(TEMPAT)}",
            f"pagi hari",
        ])
        data.append(f"Pengguna: {pertanyaan}\nAI: Langkah 1: Menganalisis pertanyaan. Langkah 2: Mencari jawaban. Jawaban: {jawaban}.\n")
    
    # Data dialog (30%)
    dialog_count = total - reasoning_count - qa_count
    for _ in range(dialog_count):
        nama = random_choice(NAMA)
        data.append(
            f"Pengguna: Halo {nama}, apa kabar?\n"
            f"AI: Langkah 1: Menyapa {nama}. Langkah 2: Menanyakan kabar. "
            f"Jawaban: Halo {nama}! Kabar saya baik. Bagaimana dengan Anda?\n"
        )
    
    random.shuffle(data)
    return data


def save_data(data: List[str], output_file: str):
    """Save data ke file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print statistik
    print(f"\n📊 Statistik Data:")
    print(f"  Total data: {len(data)}")
    print(f"  Total karakter: {sum(len(d) for d in data):,}")
    print(f"  Rata-rata panjang: {sum(len(d) for d in data) / len(data):.1f} karakter")
    
    # Contoh data
    print(f"\n📝 Contoh Data (3 random):")
    samples = random.sample(data, min(3, len(data)))
    for i, sample in enumerate(samples, 1):
        preview = sample[:150] + "..." if len(sample) > 150 else sample
        print(f"  {i}. {preview}")
        print()


# ============================================================
# MAIN
# ============================================================
def main():
    # Parse arguments
    if len(sys.argv) >= 2:
        try:
            total = int(sys.argv[1])
        except ValueError:
            total = CONFIG['total_data']
    else:
        total = CONFIG['total_data']
    
    output_file = sys.argv[2] if len(sys.argv) >= 3 else CONFIG['output_file']
    
    print("="*60)
    print("🚀 DATA GENERATOR UNTUK MINIGPT")
    print("="*60)
    print(f"  Total data target: {total}")
    print(f"  Output file      : {output_file}")
    print(f"  Random seed      : {CONFIG['seed']}")
    print(f"  Min langkah     : {CONFIG['min_langkah']}")
    print(f"  Max langkah     : {CONFIG['max_langkah']}")
    print("="*60)
    
    # Generate data
    print("\n⏳ Menghasilkan data...")
    data = generate_mixed_data(total)
    
    # Save
    print(f"\n💾 Menyimpan ke {output_file}...")
    save_data(data, output_file)
    
    print("\n✅ Selesai!")
    print(f"\n💡 Tips: Untuk training, gunakan:")
    print(f"  python3 training.py {output_file}")
    print()

if __name__ == "__main__":
    main()