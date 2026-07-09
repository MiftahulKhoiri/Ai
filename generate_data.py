import json
import random
import sys

def generate_synthetic_data(num_sentences):
    subjects = ["Saya", "Dia", "Mereka", "Kami", "Kucing", "Anjing", "Burung", "Ibu", "Ayah", "Adik", "Kakak"]
    verbs = ["makan", "minum", "tidur", "berlari", "membaca", "menulis", "memasak", "menyanyi", "melihat", "mendengar"]
    objects = ["nasi", "air", "buku", "surat", "kue", "roti", "susu", "lagu", "film", "teman"]
    adverbs = ["di rumah", "di taman", "dengan gembira", "setiap hari", "pada pagi hari", "di sekolah", "di pasar", "dengan cepat", "sambil tersenyum", "dengan malas"]
    sentences = []
    for _ in range(num_sentences):
        choice = random.random()
        if choice < 0.5:
            s = random.choice(subjects)
            v = random.choice(verbs)
            o = random.choice(objects)
            adv = random.choice(adverbs)
            sentence = f"{s} {v} {o} {adv}."
        elif choice < 0.8:
            s = random.choice(subjects)
            v = random.choice(verbs)
            adv = random.choice(adverbs)
            sentence = f"{s} {v} {adv}."
        else:
            s = random.choice(subjects)
            v = random.choice(verbs)
            o = random.choice(objects)
            sentence = f"{s} {v} {o}."
        sentences.append(sentence)
    return sentences

if __name__ == "__main__":
    num = 1000
    output = "data.json"
    if len(sys.argv) >= 2:
        num = int(sys.argv[1])
    if len(sys.argv) >= 3:
        output = sys.argv[2]
    data = generate_synthetic_data(num)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Berhasil membuat {num} kalimat sintetis -> {output}")