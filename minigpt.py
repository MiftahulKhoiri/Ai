"""
Transformer (Mini-GPT) + Tokenizer Byte-Level BPE - 100% Dibuat Sendiri (v3)
==============================================================================
Tidak menggunakan numpy, pytorch, tensorflow, transformers, atau library
eksternal apapun. Hanya modul bawaan Python: math, random, re, json,
collections, gc.

--- Perbaikan v2 (review putaran pertama) ---
 1. LayerNorm memakai Value.sqrt() khusus, bukan **0.5.
 2. Embedding & positional embedding mengembalikan node baru (identity op)
    tiap dipanggil, bukan objek Value yang sama dipakai berulang.
 3. Dropout ditambahkan pada embedding, attention, dan feedforward.
 4. <bos>/<eos> dipakai secara eksplisit saat menyiapkan data training.
 5. Loss memakai pasangan input_ids/target_ids yang eksplisit (dipisah lewat
    build_dataset), bukan indexing target[i+1] di tempat.
 6. Ada batching sungguhan (gradient accumulation antar contoh dalam batch).
 7. Optimizer AdamW ditambahkan (selain SGD).
 8. Padding mask ditambahkan di attention & loss (selain causal mask).
 9. Weight tying: embedding.weight == lm_head.weight.
10. Tokenizer diganti jadi byte-level BPE (seperti GPT-2): tidak pernah
    <unk>, decode merekonstruksi bytes UTF-8 asli sehingga spasi antar kata
    tidak pernah rusak. Positional encoding jadi learned embedding.

--- Perbaikan v3 (review putaran kedua, permintaan saat ini) ---
11. KV CACHE saat inference: MultiHeadSelfAttention.forward_incremental(),
    TransformerBlock.forward_incremental(), dan MiniGPT.forward_incremental()
    + init_cache() memproses HANYA satu token baru per langkah generate,
    bukan mengulang forward dari posisi 0 setiap kali (lihat kelas KVCache
    dan fungsi generate()).
12. Algoritma BYTE-LEVEL BPE ENCODE kini benar-benar mengikuti ranking merge
    GPT-2: ByteLevelBPETokenizer._apply_bpe() mencari pasangan dengan RANK
    TERENDAH yang ada di kata pada SETIAP langkah (bukan satu kali jalan
    berurutan lewat merge_order), karena merge di satu tempat bisa
    memunculkan pasangan baru berank lebih rendah yang harus dicoba duluan.
13. GRADIENT CLIPPING: clip_grad_norm() membatasi norma-L2 gabungan semua
    gradien sebelum optimizer.step(), mencegah exploding gradient.
14. SCHEDULER warmup + cosine decay: WarmupCosineScheduler menaikkan lr
    linear di awal lalu menurunkannya mengikuti kurva cosine.
15. SAVE/LOAD CHECKPOINT: save_checkpoint()/load_checkpoint() menyimpan
    bobot model, momen optimizer (Adam m/v/t), langkah scheduler, dan
    vocab+merges tokenizer ke satu file JSON, lalu bisa dimuat ke model baru
    dengan arsitektur yang sama persis.

Catatan performa (tetap berlaku): karena autograd berbasis skalar Python
murni, dimensi model (d_model, n_layers, seq_len) sengaja dibuat kecil agar
demo tetap selesai dalam waktu wajar. Untuk model besar, gunakan PyTorch.
"""

import math
import random
import re
import json
import gc
from collections import defaultdict, Counter


# ============================================================
# 1. ENGINE AUTOGRAD
# ============================================================
class Value:
    """Skalar yang melacak riwayat operasinya untuk backpropagation otomatis."""

    __slots__ = ('data', 'grad', '_backward', '_prev', '_op')

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "hanya mendukung pangkat konstanta"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = max(min(self.data, 60.0), -60.0)
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = max(self.data, 1e-12)
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def sqrt(self):
        # operasi sqrt khusus (lebih murah & lebih stabil daripada **0.5)
        x = max(self.data, 1e-12)
        r = math.sqrt(x)
        out = Value(r, (self,), 'sqrt')

        def _backward():
            self.grad += (0.5 / r) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def gelu(self):
        c = 0.7978845608028654  # sqrt(2/pi)
        inner = (self + (self ** 3) * 0.044715) * c
        return self * (inner.tanh() + 1.0) * 0.5

    def __neg__(self):
        return self * -1
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __rmul__(self, other):
        return self * other
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ============================================================
# 2. TOKENIZER BYTE-LEVEL BPE (SEPERTI GPT-2) DARI NOL
# ============================================================
def _bytes_to_unicode():
    """Peta tiap nilai byte (0-255) ke satu karakter unicode unik yang aman
    dicetak. Menjamin SEMUA byte punya representasi -> tidak pernah <unk>."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

# regex pra-tokenisasi ala GPT-2: pisahkan kontraksi umum, kata (dgn spasi
# di depan tetap menempel), angka, tanda baca, dan spasi berlebih.
_PRETOKEN_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']


class ByteLevelBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = {}
        self.merge_order = []
        self.merge_rank = {}   # (a, b) -> ranking (semakin kecil = semakin prioritas)

    def _text_to_symbol_words(self, text):
        words = []
        for m in _PRETOKEN_PAT.findall(text):
            byte_seq = m.encode('utf-8')
            words.append([BYTE_ENCODER[b] for b in byte_seq])
        return words

    def train(self, corpus, vocab_size=400):
        words = self._text_to_symbol_words(corpus)
        freq = Counter(tuple(w) for w in words)
        splits = {k: list(k) for k in freq}

        base_vocab = set(BYTE_ENCODER.values())  # 256 simbol byte dasar (selalu lengkap)
        merges_vocab = set()

        num_merges = max(0, vocab_size - len(base_vocab) - len(SPECIAL_TOKENS))
        for _ in range(num_merges):
            pair_counts = defaultdict(int)
            for word, f in freq.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += f
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break
            merged = ''.join(best_pair)
            self.merges[best_pair] = merged
            self.merge_order.append(best_pair)
            merges_vocab.add(merged)

            for word in splits:
                syms, new_syms, i = splits[word], [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best_pair:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                splits[word] = new_syms

        all_tokens = SPECIAL_TOKENS + sorted(base_vocab) + sorted(merges_vocab)
        self.vocab = {t: i for i, t in enumerate(all_tokens)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merge_order)}

    def _apply_bpe(self, symbols):
        """Algoritma BPE encode YANG BENAR ala GPT-2: pada setiap langkah,
        cari pasangan dengan RANK TERENDAH (paling awal dipelajari saat
        training, dengan kata lain prioritas tertinggi) yang MASIH ADA di
        kata saat ini, merge SEMUA kemunculannya, lalu ulangi dari awal
        sampai tidak ada pasangan yang bisa di-merge lagi.

        Ini BUKAN sekadar satu kali jalan berurutan lewat merge_order -
        karena merge di satu tempat bisa memunculkan pasangan baru yang
        rank-nya lebih rendah (lebih prioritas) daripada merge berikutnya
        yang "harusnya" dicoba, sehingga urutan harus dievaluasi ulang
        setiap kali, persis seperti tokenizer BPE resmi GPT-2."""
        word = list(symbols)
        if len(word) <= 1:
            return word
        while True:
            best_pair, best_rank = None, None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            first, second = best_pair
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
        return word

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.vocab['<bos>'])
        for symbols in self._text_to_symbol_words(text):
            for s in self._apply_bpe(symbols):
                # praktis tidak pernah <unk> karena semua byte dasar ada di vocab
                ids.append(self.vocab.get(s, self.vocab['<unk>']))
        if add_eos:
            ids.append(self.vocab['<eos>'])
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            tok = self.inv_vocab.get(i, '')
            if tok in SPECIAL_TOKENS:
                continue
            chars.append(tok)
        byte_str = ''.join(chars)
        raw_bytes = bytes(BYTE_DECODER[c] for c in byte_str if c in BYTE_DECODER)
        return raw_bytes.decode('utf-8', errors='replace')

    def save(self, path):
        data = {'vocab': self.vocab, 'merge_order': [list(p) for p in self.merge_order]}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = {t: int(i) for t, i in data['vocab'].items()}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.merge_order = [tuple(p) for p in data['merge_order']]
        self.merges = {tuple(p): ''.join(p) for p in self.merge_order}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merge_order)}


# ============================================================
# 3. UTILITAS VEKTOR/MATRIKS
# ============================================================
def buat_vektor(n, scale=0.1):
    return [Value(random.uniform(-scale, scale)) for _ in range(n)]


def buat_matriks(rows, cols, scale=0.1):
    return [[Value(random.uniform(-scale, scale)) for _ in range(cols)] for _ in range(rows)]


def dot(a, b):
    s = Value(0.0)
    for x, y in zip(a, b):
        s = s + x * y
    return s


def softmax(xs):
    m = max(x.data for x in xs)
    exps = [(x - m).exp() for x in xs]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


# ============================================================
# 4. DROPOUT
# ============================================================
class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        out = []
        for v in x:
            if random.random() < keep:
                out.append(v / keep)   # inverted dropout: skala saat training
            else:
                out.append(v * 0.0)    # gradien untuk unit ini jadi 0
        return out


# ============================================================
# 5. BLOK-BLOK ARSITEKTUR TRANSFORMER
# ============================================================
class KVCache:
    """Menyimpan K,V (vektor penuh d_model, belum dipecah per head) untuk
    tiap posisi yang sudah diproses, satu KVCache per TransformerBlock.
    Ini yang membuat generate() tidak perlu mengulang forward pass dari
    posisi 0 setiap kali menambah satu token baru - hanya token BARU yang
    diproses, lalu K/V-nya ditambahkan ke cache."""

    def __init__(self):
        self.K = []
        self.V = []

    def append(self, k, v):
        self.K.append(k)
        self.V.append(v)

    def reset(self):
        self.K = []
        self.V = []

    def __len__(self):
        return len(self.K)


class Linear:
    def __init__(self, n_in, n_out, bias=True):
        self.W = buat_matriks(n_out, n_in, scale=1.0 / math.sqrt(n_in))
        self.b = [Value(0.0) for _ in range(n_out)] if bias else None

    def __call__(self, x):
        out = [dot(row, x) for row in self.W]
        if self.b:
            out = [o + b for o, b in zip(out, self.b)]
        return out

    def parameters(self):
        p = [w for row in self.W for w in row]
        if self.b:
            p += self.b
        return p


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = [Value(1.0) for _ in range(dim)]
        self.beta = [Value(0.0) for _ in range(dim)]
        self.eps = eps

    def __call__(self, x):
        n = len(x)
        mean = sum(x, Value(0.0)) / n
        var = sum(((xi - mean) * (xi - mean) for xi in x), Value(0.0)) / n
        std = (var + self.eps).sqrt()          # <-- pakai sqrt() khusus, bukan **0.5
        normed = [(xi - mean) / std for xi in x]
        return [normed[i] * self.gamma[i] + self.beta[i] for i in range(n)]

    def parameters(self):
        return self.gamma + self.beta


class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads, dropout=0.1):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.Wq = Linear(d_model, d_model, bias=False)
        self.Wk = Linear(d_model, d_model, bias=False)
        self.Wv = Linear(d_model, d_model, bias=False)
        self.Wo = Linear(d_model, d_model, bias=False)
        self.drop = Dropout(dropout)

    def __call__(self, X, pad_mask=None):
        # X: seq_len x d_model. pad_mask: list 0/1 (0 = posisi padding, diabaikan)
        seq_len = len(X)
        if pad_mask is None:
            pad_mask = [1] * seq_len
        Q = [self.Wq(x) for x in X]
        K = [self.Wk(x) for x in X]
        V = [self.Wv(x) for x in X]
        scale = 1.0 / math.sqrt(self.d_head)
        outputs = [[] for _ in range(seq_len)]

        for h in range(self.n_heads):
            s, e = h * self.d_head, (h + 1) * self.d_head
            Qh = [q[s:e] for q in Q]
            Kh = [k[s:e] for k in K]
            Vh = [v[s:e] for v in V]
            for i in range(seq_len):
                # causal mask (j <= i) + padding mask (pad_mask[j] == 1)
                valid_js = [j for j in range(i + 1) if pad_mask[j] == 1]
                if not valid_js:
                    valid_js = [i]
                scores = [dot(Qh[i], Kh[j]) * scale for j in valid_js]
                weights = softmax(scores)
                head_out = [Value(0.0) for _ in range(self.d_head)]
                for w, j in zip(weights, valid_js):
                    for d in range(self.d_head):
                        head_out[d] = head_out[d] + w * Vh[j][d]
                outputs[i].extend(head_out)

        outputs = [self.Wo(o) for o in outputs]
        outputs = [self.drop(o) for o in outputs]
        return outputs

    def forward_incremental(self, x, cache):
        """Versi inkremental untuk inference dengan KV cache: x adalah SATU
        vektor (posisi token yang baru saja masuk), cache menyimpan K,V dari
        semua posisi sebelumnya milik layer ini. Tidak ada masking eksplisit
        yang diperlukan karena cache hanya berisi posisi <= sekarang (causal
        otomatis dari cara cache diisi)."""
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        cache.append(k, v)   # posisi sekarang ikut disertakan (self-attention ke diri sendiri)

        scale = 1.0 / math.sqrt(self.d_head)
        out = []
        for h in range(self.n_heads):
            s, e = h * self.d_head, (h + 1) * self.d_head
            qh = q[s:e]
            scores = [dot(qh, kk[s:e]) * scale for kk in cache.K]
            weights = softmax(scores)
            head_out = [Value(0.0) for _ in range(self.d_head)]
            for w, vv in zip(weights, cache.V):
                vh = vv[s:e]
                for d in range(self.d_head):
                    head_out[d] = head_out[d] + w * vh[d]
            out.extend(head_out)

        out = self.Wo(out)
        out = self.drop(out)   # otomatis tidak aktif saat model.eval()
        return out

    def parameters(self):
        p = []
        for layer in (self.Wq, self.Wk, self.Wv, self.Wo):
            p += layer.parameters()
        return p

    def set_training(self, mode):
        self.drop.training = mode


class FeedForward:
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.drop = Dropout(dropout)

    def __call__(self, x):
        h = [v.gelu() for v in self.fc1(x)]
        h = self.drop(h)
        return self.fc2(h)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def set_training(self, mode):
        self.drop.training = mode


class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def __call__(self, X, pad_mask=None):
        attn_out = self.attn([self.ln1(x) for x in X], pad_mask=pad_mask)
        X = [[a + b for a, b in zip(x, o)] for x, o in zip(X, attn_out)]
        ff_out = [self.ff(self.ln2(x)) for x in X]
        X = [[a + b for a, b in zip(x, o)] for x, o in zip(X, ff_out)]
        return X

    def forward_incremental(self, x, cache):
        # x: satu vektor d_model (posisi token baru). cache: KVCache milik block ini.
        attn_out = self.attn.forward_incremental(self.ln1(x), cache)
        x = [a + b for a, b in zip(x, attn_out)]
        ff_out = self.ff(self.ln2(x))
        x = [a + b for a, b in zip(x, ff_out)]
        return x

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.ff.parameters())

    def set_training(self, mode):
        self.attn.set_training(mode)
        self.ff.set_training(mode)


class Embedding:
    def __init__(self, vocab_size, d_model, scale=0.02):
        self.table = [buat_vektor(d_model, scale=scale) for _ in range(vocab_size)]

    def __call__(self, idx):
        # identity op (+0.0) -> tiap pemanggilan menghasilkan NODE GRAPH BARU.
        # Gradien dari kemunculan token yang sama tetap terakumulasi dengan
        # benar ke baris weight yang sama (ini memang perilaku embedding yang
        # benar, bukan bug) - identity op di sini murni untuk kebersihan graph.
        return [v + 0.0 for v in self.table[idx]]

    def parameters(self):
        return [p for row in self.table for p in row]


class PositionalEmbedding:
    """Learned position embedding (gaya GPT modern), menggantikan sinusoidal."""

    def __init__(self, max_len, d_model, scale=0.02):
        self.table = [buat_vektor(d_model, scale=scale) for _ in range(max_len)]

    def __call__(self, pos):
        return [v + 0.0 for v in self.table[pos]]

    def parameters(self):
        return [p for row in self.table for p in row]


# ============================================================
# 6. MODEL LENGKAP: MINI-GPT
# ============================================================
class MiniGPT:
    def __init__(self, vocab_size, d_model=16, n_heads=2, n_layers=2, d_ff=32,
                 max_len=64, dropout=0.1):
        self.d_model = d_model
        self.max_len = max_len
        self.embed = Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len, d_model)
        self.embed_drop = Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)
        self.head.W = self.embed.table   # <-- WEIGHT TYING: lm_head.weight == embedding.weight

    def __call__(self, token_ids, pad_mask=None):
        X = []
        for pos, tid in enumerate(token_ids):
            emb = self.embed(tid)
            pe = self.pos_embed(pos)
            X.append([emb[d] + pe[d] for d in range(self.d_model)])
        X = [self.embed_drop(x) for x in X]
        for block in self.blocks:
            X = block(X, pad_mask=pad_mask)
        X = [self.ln_f(x) for x in X]
        return [self.head(x) for x in X]

    def parameters(self):
        # self.embed.parameters() sudah mencakup self.head.W karena di-tie,
        # jadi head.W TIDAK ditambahkan lagi (hindari duplikasi parameter).
        p = self.embed.parameters()
        p += self.pos_embed.parameters()
        for b in self.blocks:
            p += b.parameters()
        p += self.ln_f.parameters()
        p += self.head.b  # hanya bias head yang unik
        return p

    def train(self):
        self.embed_drop.training = True
        for b in self.blocks:
            b.set_training(True)

    def eval(self):
        self.embed_drop.training = False
        for b in self.blocks:
            b.set_training(False)

    # -------- Inference dengan KV cache --------
    def init_cache(self):
        """Panggil sekali sebelum mulai generate() untuk mengosongkan cache
        tiap layer (harus dipanggil ulang tiap sesi generate baru)."""
        self.caches = [KVCache() for _ in self.blocks]

    def forward_incremental(self, token_id, pos):
        """Proses SATU token baru di posisi `pos`, memakai & memperbarui
        self.caches. Jauh lebih murah daripada memanggil model(seluruh_ids)
        ulang dari awal setiap langkah generate."""
        if pos >= self.max_len:
            raise ValueError(
                f"posisi {pos} melebihi max_len={self.max_len}; "
                "versi ini belum mendukung sliding-window cache eviction."
            )
        emb = self.embed(token_id)
        pe = self.pos_embed(pos)
        x = [emb[d] + pe[d] for d in range(self.d_model)]
        x = self.embed_drop(x)
        for block, cache in zip(self.blocks, self.caches):
            x = block.forward_incremental(x, cache)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# 7. LOSS & OPTIMIZER
# ============================================================
def cross_entropy_loss(logits_seq, target_ids, pad_mask):
    """target_ids sudah sejajar 1:1 dengan logits_seq (tidak ada indexing +1
    di sini lagi - pemisahan input/target dilakukan di build_dataset)."""
    losses = []
    for i in range(len(logits_seq)):
        if pad_mask[i] == 0:
            continue
        probs = softmax(logits_seq[i])
        losses.append(-probs[target_ids[i]].log())
    if not losses:
        return Value(0.0)
    return sum(losses, Value(0.0)) / len(losses)


class AdamW:
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2, eps, lr, wd = self.b1, self.b2, self.eps, self.lr, self.wd
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data -= lr * (m_hat / (math.sqrt(v_hat) + eps) + wd * p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


def clip_grad_norm(params, max_norm):
    """Gradient clipping global-norm (gaya torch.nn.utils.clip_grad_norm_):
    hitung norm-L2 gabungan semua gradien, lalu jika melebihi max_norm,
    skalakan SEMUA gradien dengan faktor yang sama supaya arah gradien
    tidak berubah, hanya besarnya. Mencegah training meledak (exploding
    gradient), terutama penting untuk Transformer."""
    total_sq = sum(p.grad * p.grad for p in params)
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / total_norm
        for p in params:
            p.grad *= scale
    return total_norm


class WarmupCosineScheduler:
    """Learning-rate scheduler standar gaya GPT: naik linear selama
    `warmup_steps` langkah pertama (warmup), lalu turun mengikuti kurva
    cosine dari base_lr ke min_lr sampai `total_steps`."""

    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.optimizer.lr = lr
        return lr


# ============================================================
# 8. PERSIAPAN DATASET: BOS/EOS, SLIDING WINDOW, PADDING, BATCH
# ============================================================
def build_dataset(list_of_token_ids, seq_len, pad_id):
    """list_of_token_ids: list kalimat, MASING-MASING SUDAH mengandung
    <bos> ... <eos> di ujungnya. Menghasilkan contoh (input, target, mask)
    dengan input/target dipisah EKSPLISIT (bukan indeks +1), dipad ke seq_len."""
    examples = []
    for ids in list_of_token_ids:
        for start in range(0, max(1, len(ids) - 1), seq_len):
            chunk = ids[start:start + seq_len + 1]
            if len(chunk) < 2:
                continue
            inp, tgt = chunk[:-1], chunk[1:]
            pad_len = seq_len - len(inp)
            mask = [1] * len(inp) + [0] * pad_len
            inp = inp + [pad_id] * pad_len
            tgt = tgt + [pad_id] * pad_len
            examples.append((inp, tgt, mask))
    return examples


def iter_batches(examples, batch_size, shuffle=True):
    idxs = list(range(len(examples)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        yield [examples[i] for i in idxs[start:start + batch_size]]


def train_batch(model, optimizer, batch, scheduler=None, max_grad_norm=1.0):
    optimizer.zero_grad()
    total_loss = 0.0
    for inp, tgt, mask in batch:
        logits = model(inp, pad_mask=mask)
        loss = cross_entropy_loss(logits, tgt, mask)
        (loss / len(batch)).backward()   # akumulasi gradien antar contoh dalam batch
        total_loss += loss.data
    grad_norm = clip_grad_norm(optimizer.params, max_grad_norm)  # <-- gradient clipping
    if scheduler is not None:
        scheduler.step()   # update learning rate (warmup + cosine decay) sebelum step
    optimizer.step()
    return total_loss / len(batch), grad_norm


# ============================================================
# 9. SAVE / LOAD CHECKPOINT
# ============================================================
def save_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None, config=None):
    """Simpan seluruh state (bobot model, momen optimizer, langkah
    scheduler, vocab+merges tokenizer, dan config arsitektur) ke satu file
    JSON. Urutan model.parameters() bersifat deterministik selama arsitektur
    tidak berubah, jadi cukup simpan list angka float saja per parameter."""
    data = {
        'config': config or {},
        'model_params': [p.data for p in model.parameters()],
    }
    if optimizer is not None:
        data['optimizer'] = {
            'm': optimizer.m, 'v': optimizer.v, 't': optimizer.t, 'lr': optimizer.lr,
        }
    if scheduler is not None:
        data['scheduler_step'] = scheduler.step_num
    if tokenizer is not None:
        data['tokenizer'] = {
            'vocab': tokenizer.vocab,
            'merge_order': [list(p) for p in tokenizer.merge_order],
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"[checkpoint] disimpan ke {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None):
    """Muat state dari file JSON ke objek model/optimizer/scheduler/tokenizer
    yang SUDAH dibuat dengan arsitektur yang sama persis (jumlah & urutan
    parameter harus cocok, karena itu jadi patokan pemetaan bobot)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    params = model.parameters()
    saved = data['model_params']
    if len(params) != len(saved):
        raise ValueError(
            f"Arsitektur model tidak cocok dengan checkpoint: "
            f"{len(params)} parameter saat ini vs {len(saved)} di file."
        )
    for p, val in zip(params, saved):
        p.data = val

    if optimizer is not None and 'optimizer' in data:
        optimizer.m = data['optimizer']['m']
        optimizer.v = data['optimizer']['v']
        optimizer.t = data['optimizer']['t']
        optimizer.lr = data['optimizer']['lr']

    if scheduler is not None and 'scheduler_step' in data:
        scheduler.step_num = data['scheduler_step']

    if tokenizer is not None and 'tokenizer' in data:
        tokenizer.vocab = {t: int(i) for t, i in data['tokenizer']['vocab'].items()}
        tokenizer.inv_vocab = {i: t for t, i in tokenizer.vocab.items()}
        tokenizer.merge_order = [tuple(p) for p in data['tokenizer']['merge_order']]
        tokenizer.merges = {tuple(p): ''.join(p) for p in tokenizer.merge_order}
        tokenizer.merge_rank = {pair: i for i, pair in enumerate(tokenizer.merge_order)}

    print(f"[checkpoint] dimuat dari {path}")
    return data.get('config', {})


# ============================================================
# 10. GENERATE: KV CACHE + TEMPERATURE + TOP-K/TOP-P
# ============================================================
def _sample_from_logits(logits, temperature=0.9, top_k=None, top_p=0.9):
    scaled_data = [v.data / max(temperature, 1e-6) for v in logits]
    m = max(scaled_data)
    exps = [math.exp(x - m) for x in scaled_data]
    s = sum(exps)
    probs = [e / s for e in exps]

    if top_k is not None:
        keep = set(sorted(range(len(probs)), key=lambda i: -probs[i])[:top_k])
        probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]
    if top_p is not None:
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        cum, keep = 0.0, set()
        for i in order:
            cum += probs[i]
            keep.add(i)
            if cum >= top_p:
                break
        probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]

    total = sum(probs)
    if total <= 0:
        return max(range(len(probs)), key=lambda i: scaled_data[i])
    probs = [p / total for p in probs]
    r, cum, next_id = random.random(), 0.0, len(probs) - 1
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            next_id = i
            break
    return next_id


def generate(model, tokenizer, prompt, max_new_tokens=25, temperature=0.9,
             top_k=None, top_p=0.9):
    """Generate teks memakai KV CACHE: setiap token prompt diproses satu
    kali untuk mengisi cache, lalu tiap token baru yang dihasilkan HANYA
    memproses SATU token (bukan mengulang seluruh konteks dari awal seperti
    versi sebelumnya) - inilah manfaat utama KV cache saat inference."""
    model.eval()
    model.init_cache()
    ids = tokenizer.encode(prompt, add_bos=True)
    eos_id = tokenizer.vocab['<eos>']

    logits = None
    for pos, tid in enumerate(ids):
        logits = model.forward_incremental(tid, pos)   # isi cache dari prompt

    for _ in range(max_new_tokens):
        next_id = _sample_from_logits(logits, temperature, top_k, top_p)
        ids.append(next_id)
        if next_id == eos_id:
            break
        pos = len(ids) - 1
        if pos >= model.max_len:
            break  # batas context window tercapai (versi ini belum menggeser cache)
        logits = model.forward_incremental(next_id, pos)   # <-- hanya 1 token baru diproses

    model.train()
    return tokenizer.decode(ids)


# ============================================================
# 11. DEMO
# ============================================================
if __name__ == "__main__":
    random.seed(0)

    kalimat = [
        "kucing suka makan ikan.",
        "anjing suka makan tulang.",
        "kucing dan anjing adalah hewan peliharaan.",
        "burung suka terbang tinggi di langit.",
        "ikan berenang di air.",
        "kucing suka tidur di sofa.",
    ]

    print("=== Melatih Tokenizer Byte-Level BPE ===")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(" ".join(kalimat), vocab_size=320)
    print(f"Ukuran vocab: {len(tokenizer.vocab)}")

    enc = tokenizer.encode("kucing suka makan ikan 🐟")  # sertakan emoji: uji anti-<unk>
    print(f"Encode (dgn emoji) -> {enc}")
    print(f"Decode kembali     -> '{tokenizer.decode(enc)}'")

    print("\n=== Menyiapkan Dataset (BOS/EOS + input/target eksplisit) ===")
    pad_id = tokenizer.vocab['<pad>']
    token_lists = [tokenizer.encode(k, add_bos=True, add_eos=True) for k in kalimat]
    seq_len = 10
    dataset = build_dataset(token_lists, seq_len=seq_len, pad_id=pad_id)
    print(f"Jumlah contoh training: {len(dataset)}")

    print("\n=== Membangun Mini-GPT (dengan dropout, weight tying, AdamW) ===")
    # Catatan performa: autograd berbasis skalar Python murni (bukan numpy),
    # jadi dimensi sengaja dibuat kecil (d_model, d_ff, n_layers, seq_len)
    # agar demo tetap selesai dalam waktu wajar untuk belajar konsepnya.
    model = MiniGPT(vocab_size=len(tokenizer.vocab), d_model=12, n_heads=2,
                     n_layers=1, d_ff=24, max_len=32, dropout=0.1)
    model.train()
    params = model.parameters()
    print(f"Jumlah parameter (Value): {len(params)}")

    optimizer = AdamW(params, lr=0.03, weight_decay=0.01)

    epochs = 6
    batch_size = 2
    n_batches_per_epoch = math.ceil(len(dataset) / batch_size)
    total_steps = epochs * n_batches_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=max(2, total_steps // 5),
        total_steps=total_steps, base_lr=0.03, min_lr=0.002,
    )

    print("\n=== Training (batch + AdamW + gradient clipping + scheduler) ===")
    for epoch in range(1, epochs + 1):
        epoch_loss, n_batch = 0.0, 0
        for batch in iter_batches(dataset, batch_size=batch_size, shuffle=True):
            loss_val, grad_norm = train_batch(model, optimizer, batch,
                                               scheduler=scheduler, max_grad_norm=1.0)
            epoch_loss += loss_val
            n_batch += 1
        if epoch == 1 or epoch % 3 == 0:
            print(f"Epoch {epoch:2d}/{epochs} - Loss: {epoch_loss / n_batch:.4f} "
                  f"- lr: {optimizer.lr:.5f} - grad_norm terakhir: {grad_norm:.3f}")
        gc.collect()  # lepaskan graph epoch sebelumnya secara eksplisit

    print("\n=== Menyimpan Checkpoint ===")
    ckpt_path = "checkpoint_minigpt.json"
    model_config = {
        'vocab_size': len(tokenizer.vocab), 'd_model': model.d_model,
        'n_heads': 2, 'n_layers': len(model.blocks), 'd_ff': 24,
        'max_len': model.max_len, 'dropout': 0.1,
    }
    save_checkpoint(ckpt_path, model, optimizer, scheduler, tokenizer, config=model_config)

    print("\n=== Memuat Ulang Checkpoint ke Model BARU (uji save/load) ===")
    cfg = json.load(open(ckpt_path))['config']
    model2 = MiniGPT(vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
                      n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
                      d_ff=cfg['d_ff'], max_len=cfg['max_len'], dropout=cfg['dropout'])
    tokenizer2 = ByteLevelBPETokenizer()
    load_checkpoint(ckpt_path, model2, tokenizer=tokenizer2)

    print("\n=== Generate Teks dengan Model Hasil Load (KV cache aktif) ===")
    hasil = generate(model2, tokenizer2, "kucing suka", max_new_tokens=10,
                      temperature=0.8, top_p=0.9)
    print("Hasil:", hasil)