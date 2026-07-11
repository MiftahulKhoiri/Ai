"""
minigpt_utils.py - Fungsi bantuan untuk training dan dataset
Digunakan bersama modul C++ minigpt.
"""

import random
import json
import os
from typing import List, Tuple, Dict, Any, Optional

from minigpt import (
    MiniGPT, 
    ByteLevelBPETokenizer, 
    AdamW, 
    WarmupCosineScheduler,
    cross_entropy_loss,
    clip_grad_norm,
    Value
)


def build_dataset(list_of_token_ids: List[List[int]], seq_len: int, pad_id: int) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    list_of_token_ids: list kalimat, masing-masing SUDAH mengandung <bos> ... <eos>.
    Menghasilkan contoh (input, target, mask) dengan input/target dipisah eksplisit,
    di-pad ke seq_len.
    """
    examples = []
    for ids in list_of_token_ids:
        # Jika ids lebih pendek dari seq_len, pad
        if len(ids) < 2:
            continue
            
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


def iter_batches(examples: List[Tuple[List[int], List[int], List[int]]], 
                 batch_size: int, 
                 shuffle: bool = True):
    """Menghasilkan batch dari contoh."""
    idxs = list(range(len(examples)))
    if shuffle:
        random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch_indices = idxs[start:start + batch_size]
        yield [examples[i] for i in batch_indices]


def train_batch(model: MiniGPT, 
                optimizer: AdamW, 
                batch: List[Tuple[List[int], List[int], List[int]]], 
                scheduler: Optional[WarmupCosineScheduler] = None, 
                max_grad_norm: float = 1.0) -> Tuple[float, float]:
    """
    Melatih satu batch.
    model, optimizer adalah objek dari modul C++ minigpt.
    """
    optimizer.zero_grad()
    total_loss = 0.0
    n_samples = 0

    for inp, tgt, mask in batch:
        # Forward
        logits = model.forward(inp)
        
        # Compute loss
        loss = cross_entropy_loss(logits, tgt, mask)
        
        # Accumulate loss
        total_loss += loss.data
        n_samples += 1
        
        # Backward (loss dibagi batch_size untuk average)
        loss.backward()

    # Clip gradient
    params = model.parameters()
    grad_norm = clip_grad_norm(params, max_grad_norm)

    # Step optimizer
    optimizer.step()

    # Step scheduler (jika ada)
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, grad_norm


def save_checkpoint(path: str, 
                    model: MiniGPT, 
                    optimizer: Optional[AdamW] = None, 
                    scheduler: Optional[WarmupCosineScheduler] = None, 
                    tokenizer: Optional[ByteLevelBPETokenizer] = None,
                    config: Optional[Dict[str, Any]] = None,
                    total_steps: Optional[int] = None,
                    epoch: int = 0,
                    loss: float = 0.0):
    """
    Simpan checkpoint model.
    """
    data = {
        'config': config or {},
        'epoch': epoch,
        'loss': loss,
        'model_params': [p.data for p in model.parameters()],
    }
    
    # Save optimizer state
    if optimizer is not None:
        data['optimizer'] = {
            'm': optimizer.get_m(),
            'v': optimizer.get_v(),
            't': optimizer.get_t(),
            'lr': optimizer.lr,
        }
    
    # Save scheduler state
    if scheduler is not None:
        data['scheduler_step'] = scheduler.get_step_num()
    
    # Save total steps
    if total_steps is not None:
        data['total_steps'] = total_steps
    
    # Save tokenizer
    if tokenizer is not None:
        data['tokenizer'] = {
            'vocab': tokenizer.get_vocab(),
            'inv_vocab': {str(k): v for k, v in tokenizer.get_inv_vocab().items()},
            'merge_order': tokenizer.get_merge_order(),
        }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Checkpoint disimpan ke {path}")


def load_checkpoint(path: str, 
                    additional_steps: int = 0, 
                    warmup_steps: int = 30, 
                    min_lr: float = 1e-5) -> Tuple[MiniGPT, AdamW, WarmupCosineScheduler, 
                                                   ByteLevelBPETokenizer, Dict[str, Any], int]:
    """
    Muat checkpoint dan bangun ulang semua object dari config yang tersimpan.
    
    Return:
        model, optimizer, scheduler, tokenizer, config, total_steps
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = data.get('config', {})
    
    # 1. Bangun model dari config
    model = MiniGPT(
        vocab_size=config.get('vocab_size', 200),
        d_model=config.get('d_model', 4),
        n_heads=config.get('n_heads', 2),
        n_layers=config.get('n_layers', 1),
        d_ff=config.get('d_ff', 16),
        max_len=config.get('max_len', 32),
        dropout=config.get('dropout', 0.1),
    )
    
    # Restore model parameters
    params = model.parameters()
    saved_params = data.get('model_params', [])
    if len(params) != len(saved_params):
        print(f"⚠️  Warning: Parameter count mismatch: {len(params)} vs {len(saved_params)}")
        # Ambil minimum
        min_len = min(len(params), len(saved_params))
        for i in range(min_len):
            params[i].data = saved_params[i]
    else:
        for p, val in zip(params, saved_params):
            p.data = val

    # 2. Bangun optimizer
    saved_lr = data.get('optimizer', {}).get('lr', 0.01)
    optimizer = AdamW(params, lr=saved_lr, weight_decay=0.01)
    
    # Restore optimizer state
    if 'optimizer' in data:
        opt_data = data['optimizer']
        if 'm' in opt_data:
            optimizer.set_m(opt_data['m'])
        if 'v' in opt_data:
            optimizer.set_v(opt_data['v'])
        if 't' in opt_data:
            optimizer.set_t(opt_data['t'])
        if 'lr' in opt_data:
            optimizer.lr = opt_data['lr']

    # 3. Bangun scheduler
    old_step_num = data.get('scheduler_step', 0)
    old_total_steps = data.get('total_steps', old_step_num + 1)
    total_steps = old_step_num + additional_steps if additional_steps > 0 else old_total_steps

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=saved_lr,
        min_lr=min_lr,
    )
    scheduler.set_step_num(old_step_num)

    # 4. Bangun tokenizer
    tokenizer = ByteLevelBPETokenizer()
    if 'tokenizer' in data:
        tok_data = data['tokenizer']
        if 'vocab' in tok_data:
            tokenizer.set_vocab(tok_data['vocab'])
        if 'inv_vocab' in tok_data:
            inv_vocab = {int(k): v for k, v in tok_data['inv_vocab'].items()}
            tokenizer.set_inv_vocab(inv_vocab)
        if 'merge_order' in tok_data:
            tokenizer.set_merge_order(tok_data['merge_order'])

    print(f"✅ Checkpoint loaded: {path} (step={old_step_num}, total_steps={total_steps})")
    print(f"   Vocab size: {len(tokenizer.get_vocab())}")

    return model, optimizer, scheduler, tokenizer, config, total_steps


def test_model(model: MiniGPT, tokenizer: ByteLevelBPETokenizer, prompt: str = "Halo", max_tokens: int = 20):
    """Simple test generation"""
    from minigpt import generate
    
    try:
        result = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        if isinstance(result, list):
            result = result[0] if result else ""
        print(f"  Prompt: {prompt}")
        print(f"  Result: {result}")
        return result
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return ""