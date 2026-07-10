"""
minigpt_utils.py - Fungsi bantuan untuk training dan dataset
Digunakan bersama modul C++ minigpt.
"""

import random
import json
from minigpt import MiniGPT, ByteLevelBPETokenizer, AdamW, WarmupCosineScheduler


def build_dataset(list_of_token_ids, seq_len, pad_id):
    """
    list_of_token_ids: list kalimat, masing-masing SUDAH mengandung <bos> ... <eos>.
    Menghasilkan contoh (input, target, mask) dengan input/target dipisah eksplisit,
    di-pad ke seq_len.
    """
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
    """Menghasilkan batch dari contoh."""
    idxs = list(range(len(examples)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        yield [examples[i] for i in idxs[start:start + batch_size]]


def train_batch(model, optimizer, batch, scheduler=None, max_grad_norm=1.0):
    """
    Melatih satu batch.
    model, optimizer adalah objek dari modul C++ minigpt.
    """
    from minigpt import cross_entropy_loss, clip_grad_norm

    optimizer.zero_grad()
    total_loss = 0.0

    for inp, tgt, mask in batch:
        logits = model.forward(inp, pad_mask=mask)
        loss = cross_entropy_loss(logits, tgt, mask)
        # loss adalah objek Value (C++), diasumsikan mendukung pembagian
        (loss / len(batch)).backward()
        total_loss += loss.data

    grad_norm = clip_grad_norm(optimizer.params, max_grad_norm)

    if scheduler is not None:
        scheduler.step()

    optimizer.step()

    return total_loss / len(batch), grad_norm


def save_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None, config=None):
    """Simpan checkpoint model."""
    data = {
        'config': config or {},
        'model_params': [p.data for p in model.parameters()],
    }
    if optimizer is not None:
        data['optimizer'] = {
            'm': optimizer.m,
            'v': optimizer.v,
            't': optimizer.t,
            'lr': optimizer.lr,
        }
    if scheduler is not None:
        data['scheduler_step'] = scheduler.step_num
        # Simpan juga total_steps jika ada (kita tambahkan atribut ini di training)
        if hasattr(scheduler, 'total_steps'):
            data['total_steps'] = scheduler.total_steps
    if tokenizer is not None:
        data['tokenizer'] = {
            'vocab': tokenizer.vocab,
            'merge_order': tokenizer.merge_order,
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"[checkpoint] disimpan ke {path}")


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, tokenizer=None):
    """
    Memuat checkpoint.
    - Jika model=None, model akan dibuat otomatis dari config.
    - Jika optimizer/scheduler/tokenizer=None, akan dibuat jika data tersedia.
    Mengembalikan (model, optimizer, scheduler, tokenizer, config)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = data.get('config', {})

    # ---- Model ----
    if model is None:
        # Buat model baru sesuai config
        model = MiniGPT(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_len=config['max_len'],
            dropout=config.get('dropout', 0.0)
        )
    params = model.parameters()
    saved_params = data['model_params']
    if len(params) != len(saved_params):
        raise ValueError(
            f"Arsitektur model tidak cocok: "
            f"{len(params)} parameter saat ini vs {len(saved_params)} di checkpoint."
        )
    for p, val in zip(params, saved_params):
        p.data = val

    # ---- Optimizer ----
    if optimizer is None and 'optimizer' in data:
        opt_data = data['optimizer']
        optimizer = AdamW(model.parameters(), lr=opt_data.get('lr', 0.001))
        optimizer.m = opt_data['m']
        optimizer.v = opt_data['v']
        optimizer.t = opt_data['t']
    elif optimizer is not None and 'optimizer' in data:
        optimizer.m = data['optimizer']['m']
        optimizer.v = data['optimizer']['v']
        optimizer.t = data['optimizer']['t']
        optimizer.lr = data['optimizer']['lr']

    # ---- Scheduler ----
    if scheduler is None and 'scheduler_step' in data:
        total_steps = data.get('total_steps', data['scheduler_step'] + 1)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=0,
            total_steps=total_steps,
            base_lr=optimizer.lr,
            min_lr=1e-5
        )
        scheduler.step_num = data['scheduler_step']
        scheduler.total_steps = total_steps
    elif scheduler is not None and 'scheduler_step' in data:
        scheduler.step_num = data['scheduler_step']
        if 'total_steps' in data:
            scheduler.total_steps = data['total_steps']

    # ---- Tokenizer ----
    if tokenizer is None and 'tokenizer' in data:
        tok_data = data['tokenizer']
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.vocab = tok_data['vocab']
        tokenizer.inv_vocab = {int(i): t for t, i in tok_data['vocab'].items()}
        tokenizer.merge_order = tok_data['merge_order']
        tokenizer.merge_rank = {m: i for i, m in enumerate(tokenizer.merge_order)}
    elif tokenizer is not None and 'tokenizer' in data:
        tokenizer.vocab = data['tokenizer']['vocab']
        tokenizer.inv_vocab = {int(i): t for t, i in tokenizer.vocab.items()}
        tokenizer.merge_order = data['tokenizer']['merge_order']
        tokenizer.merge_rank = {m: i for i, m in enumerate(tokenizer.merge_order)}

    print(f"[checkpoint] dimuat dari {path}")
    return model, optimizer, scheduler, tokenizer, config