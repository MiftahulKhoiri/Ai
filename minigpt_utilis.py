"""
minigpt_utils.py - Fungsi bantuan untuk training dan dataset
Digunakan bersama modul C++ minigpt.
"""

import random
from minigpt import Value  # jika diperlukan


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
        # Karena loss adalah Value (C++), kita perlu memanggil backward
        # dan membagi gradien secara manual untuk batch
        (loss / len(batch)).backward()  # Ini memerlukan operator / pada Value
        total_loss += loss.data

    grad_norm = clip_grad_norm(optimizer.params, max_grad_norm)

    if scheduler is not None:
        scheduler.step()

    optimizer.step()

    return total_loss / len(batch), grad_norm


def save_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None, config=None):
    """Simpan checkpoint model."""
    import json

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
    if tokenizer is not None:
        data['tokenizer'] = {
            'vocab': tokenizer.vocab,
            'merge_order': tokenizer.merge_order,
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"[checkpoint] disimpan ke {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None):
    """Muat checkpoint model."""
    import json

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
        tokenizer.vocab = data['tokenizer']['vocab']
        tokenizer.inv_vocab = {int(i): t for t, i in tokenizer.vocab.items()}
        tokenizer.merge_order = data['tokenizer']['merge_order']
        # Rebuild merge_rank jika diperlukan (tergantung implementasi tokenizer)
        # Biasanya tokenizer.load() sudah menangani ini

    print(f"[checkpoint] dimuat dari {path}")
    return data.get('config', {})