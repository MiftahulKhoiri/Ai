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


def save_checkpoint(path, model, optimizer=None, scheduler=None, tokenizer=None,
                     config=None, total_steps=None):
    """
    Simpan checkpoint model.

    total_steps: jumlah total steps yang direncanakan untuk scheduler ini.
                 WAJIB diisi kalau ingin resume training nanti mengetahui
                 total_steps sebelumnya (karena objek WarmupCosineScheduler
                 dari C++ tidak menyimpan/mengekspos total_steps).
    """
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
    if total_steps is not None:
        data['total_steps'] = total_steps
    if tokenizer is not None:
        data['tokenizer'] = {
            'vocab': tokenizer.vocab,
            'merge_order': tokenizer.merge_order,
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"[checkpoint] disimpan ke {path}")


def load_checkpoint(path, additional_steps=0, warmup_steps=0, min_lr=1e-5):
    """
    Muat checkpoint dan BANGUN ULANG semua object (model, optimizer, scheduler,
    tokenizer) dari nol berdasarkan config yang tersimpan di file.

    Ini beda dari versi lama yang mengharuskan model/optimizer/scheduler
    sudah dibuat lebih dulu sebelum dipanggil. Sekarang cukup panggil dengan
    path saja.

    Parameter:
        path            : path file checkpoint (.json)
        additional_steps: jumlah step tambahan untuk sesi training lanjutan.
                           Jika 0, total_steps lama (dari file) dipakai apa adanya.
        warmup_steps    : warmup_steps baru untuk scheduler yang dibangun ulang.
        min_lr          : min_lr baru untuk scheduler yang dibangun ulang.

    Return:
        model, optimizer, scheduler, tokenizer, config, total_steps
    """
    import json
    from minigpt import MiniGPT, AdamW, WarmupCosineScheduler, ByteLevelBPETokenizer

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = data['config']

    # 1. Bangun model dari config, lalu isi bobot dari checkpoint
    model = MiniGPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout'],
    )
    params = model.parameters()
    saved = data['model_params']
    if len(params) != len(saved):
        raise ValueError(
            f"Arsitektur model tidak cocok dengan checkpoint: "
            f"{len(params)} parameter saat ini vs {len(saved)} di file."
        )
    for p, val in zip(params, saved):
        p.data = val

    # 2. Bangun optimizer dan pulihkan state Adam
    if 'optimizer' not in data:
        raise ValueError("Checkpoint tidak memiliki state optimizer.")
    saved_lr = data['optimizer']['lr']
    optimizer = AdamW(params, lr=saved_lr)
    optimizer.m = data['optimizer']['m']
    optimizer.v = data['optimizer']['v']
    optimizer.t = data['optimizer']['t']

    # 3. Bangun scheduler baru dengan total_steps yang sesuai, lalu pulihkan step_num
    if 'scheduler_step' not in data:
        raise ValueError("Checkpoint tidak memiliki scheduler_step.")
    old_step_num = data['scheduler_step']
    old_total_steps = data.get('total_steps', old_step_num)
    total_steps = old_step_num + additional_steps if additional_steps else old_total_steps

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=saved_lr,
        min_lr=min_lr,
    )
    scheduler.step_num = old_step_num

    # 4. Bangun tokenizer dan pulihkan vocab
    if 'tokenizer' not in data:
        raise ValueError("Checkpoint tidak memiliki data tokenizer.")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.vocab = data['tokenizer']['vocab']
    tokenizer.inv_vocab = {int(i): t for t, i in tokenizer.vocab.items()}
    tokenizer.merge_order = data['tokenizer']['merge_order']

    print(f"[checkpoint] dimuat dari {path} "
          f"(step terakhir={old_step_num}, total_steps baru={total_steps})")

    return model, optimizer, scheduler, tokenizer, config, total_steps
