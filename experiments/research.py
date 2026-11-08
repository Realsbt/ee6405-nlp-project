"""Explicit, ablatable A4000 model and optimizer extensions."""
import re
import torch
from torch import nn


def validate(config):
    depth = config['architecture']['unfreeze_bert_layers']
    if config['architecture'].get('max_seq_length') != 512:
        raise ValueError('The frozen protocol requires a 512-token window')
    if type(depth) is not int or depth not in (1, 2, 4, 6, 12):
        raise ValueError('unfreeze_bert_layers must be 1/2/4/6/12')
    if config.get('unfreeze_embeddings', False) and depth != 12:
        raise ValueError('Embeddings may only be unfrozen with all 12 layers')
    if config['batch_size'] % 4 or (config['use_supcon'] and config['batch_size'] < 8):
        raise ValueError('SupCon requires balanced real batch >=8, divisible by four')
    if type(config.get('gradient_accumulation_steps', 1)) is not int or config.get('gradient_accumulation_steps', 1) < 1:
        raise ValueError('gradient_accumulation_steps must be a positive integer')
    if config.get('attention_implementation', 'sdpa') not in ('sdpa', 'eager'):
        raise ValueError('attention_implementation must be sdpa or eager')
    if not 0 < config.get('layer_lr_decay', 1) <= 1:
        raise ValueError('layer_lr_decay must be in (0,1]')
    for name in ('aux_content_weight', 'aux_edit_weight'):
        if config.get(name, 0) < 0:
            raise ValueError(f'{name} must be nonnegative')
    for name in ('learning_rate', 'head_learning_rate', 'embedding_learning_rate'):
        if name in config and config[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if config.get('torch_compile', False) and config['model_type'] != 'roberta':
        raise ValueError('torch_compile is supported for roberta only')
    if config.get('embedding_learning_rate', 0) > 2.5e-6:
        raise ValueError('Embedding LR must not exceed 2.5e-6')
    if config.get('dataloader_num_workers', 0) < 0:
        raise ValueError('dataloader_num_workers must be nonnegative')


def configure_model(model, config):
    validate(config)
    attention = config.get('attention_implementation')
    if attention and model.bert_model.attn_implementation != attention:
        from transformers import AutoModel
        model.bert_model = AutoModel.from_pretrained(
            model.config['backbone_model_path'], attn_implementation=attention, local_files_only=True)
    depth = config['architecture']['unfreeze_bert_layers']
    model.bert_model.requires_grad_(False)
    for layer in model.bert_model.encoder.layer[-depth:]:
        layer.requires_grad_(True)
    model.bert_model.embeddings.requires_grad_(config.get('unfreeze_embeddings', False))
    if config.get('gradient_checkpointing', False):
        model.bert_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    dim = config['architecture']['bert_dim']
    if config.get('supcon_projection', False):
        if config['model_type'] != 'roberta':
            raise ValueError('Research projection/auxiliary heads require roberta')
        model.supcon_projection = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, 128))
    for key, name in (('aux_content_weight', 'content_head'), ('aux_edit_weight', 'edit_head')):
        if config.get(key, 0):
            if config['model_type'] != 'roberta':
                raise ValueError('Auxiliary heads require roberta')
            setattr(model, name, nn.Linear(dim, 2))
    return model


def optimizer_groups(model, config):
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        match = re.match(r'bert_model.encoder.layer.(\d+)\.', name)
        if match:
            idx = int(match[1]); group = f'encoder_{idx}'
            lr = config['learning_rate'] * config.get('layer_lr_decay', 1) ** (11 - idx)
        elif name.startswith('bert_model.embeddings.'):
            group = 'embeddings'; lr = config.get('embedding_learning_rate', 2e-6)
        else:
            group = name.split('.')[0]; lr = config.get('head_learning_rate', config['learning_rate'])
        decay = not (name.endswith('.bias') or 'LayerNorm' in name)
        # Preserve original decay when no discriminative recipe is requested.
        wd = config['weight_decay'] if decay or not config.get('exclude_norm_bias_decay', False) else 0.0
        key = (group, lr, wd)
        buckets.setdefault(key, []).append(p)
    return [{'params': params, 'lr': lr, 'weight_decay': wd, 'name': name}
            for (name, lr, wd), params in buckets.items()]


def auxiliary_loss(model, features, labels, config):
    loss = features.new_zeros((), dtype=torch.float32)
    for key, name, target in (
        ('aux_content_weight', 'content_head', (labels >= 2).long()),
        ('aux_edit_weight', 'edit_head', labels.remainder(2)),
    ):
        if config.get(key, 0):
            loss = loss + config[key] * nn.functional.cross_entropy(getattr(model, name)(features).float(), target)
    return loss
