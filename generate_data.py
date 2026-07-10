import json
import random
import sys

def generate_reasoning_data(num_sentences):
    """
    Menghasilkan data latih dengan format:
    "Pengguna: <pertanyaan>\nAI: Langkah 1: ... Langkah 2: ... Jawaban: <jawaban>\n"

    Jenis pertanyaan: matematika sederhana, pengetahuan, logika, dll.
    """
    data = []

    # Template pertanyaan dan cara menjawab dengan langkah-langkah
    templates = [
        # --- Penjumlahan ---
        lambda a, b: (
            f"Pengguna: Berapa {a} + {b}?\n"
            f"AI: Langkah 1: Saya perlu menjumlahkan {a} dan {b}. "
            f"Langkah 2: {a} + {b} = {a+b}. "
            f"Jawaban: {a+b}.\n"
        ),
        # --- Pengurangan ---
        lambda a, b: (
            f"Pengguna: Berapa {a} - {b}?\n"
            f"AI: Langkah 1: Saya perlu mengurangkan {b} dari {a}. "
            f"Langkah 2: {a} - {b} = {a-b}. "
            f"Jawaban: {a-b}.\n"
        ),
        # --- Perkalian ---
        lambda a, b: (
            f"Pengguna: Berapa {a} × {b}?\n"
            f"AI: Langkah 1: Saya perlu mengalikan {a} dengan {b}. "
            f"Langkah 2: {a} × {b} = {a*b}. "
            f"Jawaban: {a*b}.\n"
        ),
        # --- Pengetahuan sederhana (warna, hewan, dll.) ---
        lambda obj, warna: (
            f"Pengguna: Apa warna {obj}?\n"
            f"AI: Langkah 1: Saya ingat bahwa {obj} biasanya berwarna {warna}. "
            f"Jawaban: {warna}.\n"
        ),
        # --- Klasifikasi hewan ---
        lambda hewan, suara: (
            f"Pengguna: Suara apa yang dihasilkan {hewan}?\n"
            f"AI: Langkah 1: {hewan} adalah hewan yang bersuara '{suara}'. "
            f"Jawaban: {suara}.\n"
        ),
        # --- Logika sederhana (lebih besar/kecil) ---
        lambda a, b: (
            f"Pengguna: Mana yang lebih besar, {a} atau {b}?\n"
            f"AI: Langkah 1: Membandingkan {a} dan {b}. "
            f"Langkah 2: {max(a,b)} lebih besar dari {min(a,b)}. "
            f"Jawaban: {max(a,b)}.\n"
        ),
        # --- Fakta sederhana ---
        lambda: (
            f"Pengguna: Berapa jumlah hari dalam seminggu?\n"
            f"AI: Langkah 1: Satu minggu terdiri dari 7 hari. "
            f"Jawaban: 7.\n"
        ),
        # --- Kombinasi cerita pendek ---
        lambda nama, tempat: (
            f"Pengguna: Kemana {nama} pergi?\n"
            f"AI: Langkah 1: Berdasarkan cerita, {nama} pergi ke {tempat}. "
            f"Jawaban: {tempat}.\n"
        ),
    ]

    # Pool nilai untuk slot template
    angka_kecil = list(range(1, 21))
    objek_warna = [("apel", "merah"), ("langit", "biru"), ("daun", "hijau"), ("pisang", "kuning"), ("tanah", "coklat")]
    hewan_suara = [("kucing", "meong"), ("anjing", "gukguk"), ("sapi", "moo"), ("domba", "mbek"), ("ayam", "kukuruyuk")]
    nama_tempat = [("Andi", "pasar"), ("Budi", "sekolah"), ("Cici", "taman"), ("Dedi", "rumah sakit")]

    for _ in range(num_sentences):
        choice = random.random()
        if choice < 0.3:  # 30% aritmatika
            op = random.choice(['+', '-', '*'])
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            if op == '+':
                sent = templates[0](a, b)
            elif op == '-':
                if a < b:
                    a, b = b, a  # pastikan hasil positif
                sent = templates[1](a, b)
            else:
                sent = templates[2](a, b)
        elif choice < 0.5:  # 20% warna
            obj, warna = random.choice(objek_warna)
            sent = templates[3](obj, warna)
        elif choice < 0.7:  # 20% suara hewan
            hewan, suara = random.choice(hewan_suara)
            sent = templates[4](hewan, suara)
        elif choice < 0.85: # 15% lebih besar
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            if a == b:
                b = a + 1
            sent = templates[5](a, b)
        elif choice < 0.95: # 10% fakta tetap
            sent = templates[6]()
        else:               # 5% cerita
            nama, tempat = random.choice(nama_tempat)
            sent = templates[7](nama, tempat)

        data.append(sent)

    return data

if __name__ == "__main__":
    num = 460
    output = "data.json"
    if len(sys.argv) >= 2:
        num = int(sys.argv[1])
    if len(sys.argv) >= 3:
        output = sys.argv[2]
    
    print(f"Menghasilkan {num} data reasoning...")
    data = generate_reasoning_data(num)
    
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Berhasil membuat {num} data reasoning -> {output}")
    print("\nContoh data:")
    for i, d in enumerate(data[:3], 1):
        print(f"{i}. {d[:100]}...")