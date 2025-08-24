"""
Core feature importance extraction for FT-Transformer vs Sparse FTT+
Focus on the essential: attention maps extraction and permutation test
"""

import math
import typing as ty
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
import json
from tqdm import tqdm

from rtdl_revisiting_models import lib


class AttentionMapExtractor:
    """Extracteur d'importance basé sur les cartes d'attention"""
    
    def __init__(self):
        self.attention_maps: ty.List[torch.Tensor] = []
        
    def __call__(self, module, input, output):
        """Hook pour capturer les cartes d'attention"""
        if isinstance(output, tuple) and len(output) >= 2:
            attention = output[1]
            if hasattr(attention, 'shape'):
                self.attention_maps.append(attention.detach().cpu())
    
    def reset(self):
        self.attention_maps.clear()


class SparseAttentionExtractor:
    """Extracteur pour Sparse FTT+ avec attention interprétable"""
    
    def __init__(self):
        self.attention_maps: ty.List[torch.Tensor] = []
        
    def __call__(self, module, input, output):
        """Hook spécialisé pour InterpretableMultiHeadAttention"""
        if isinstance(output, tuple) and len(output) >= 2:
            attention = output[1]
            self.attention_maps.append(attention.detach().cpu())
    
    def reset(self):
        self.attention_maps.clear()


def extract_ft_transformer_importance(model, X_num, X_cat):
    """Extrait l'importance pour FT-Transformer (méthode des auteurs)"""
    hook = AttentionMapExtractor()
    hooks = []
    
    def patched_forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        """Forward patché pour capturer l'attention"""
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]
        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = attention @ v
        output = output.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        if self.W_out is not None:
            output = self.W_out(output)
        return output, attention
    
    original_forwards = []
    for layer in model.layers:
        attention_module = layer['attention']
        original_forwards.append(attention_module.forward)
        attention_module.forward = patched_forward.__get__(attention_module)
        h = attention_module.register_forward_hook(hook)
        hooks.append(h)
    
    model.eval()
    with torch.no_grad():
        batch_size = 128
        n_samples = X_num.shape[0] if X_num is not None else X_cat.shape[0]
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_num_batch = X_num[i:end_idx] if X_num is not None else None
            X_cat_batch = X_cat[i:end_idx] if X_cat is not None else None
            _ = model(X_num_batch, X_cat_batch)
    
    for h in hooks:
        h.remove()
    for layer, original_forward in zip(model.layers, original_forwards):
        layer['attention'].forward = original_forward
    
    if not hook.attention_maps:
        raise ValueError("No attention maps collected")
    
    attention_maps = torch.cat(hook.attention_maps)
    n_objects = n_samples
    n_blocks = len(model.layers)
    n_heads = model.layers[0]['attention'].n_heads
    n_features = (0 if X_num is None else X_num.shape[1]) + (0 if X_cat is None else X_cat.shape[1])
    n_tokens = n_features + 1
    
    expected_shape = (n_objects * n_blocks * n_heads, n_tokens, n_tokens)
    assert attention_maps.shape == expected_shape, f"Expected {expected_shape}, got {attention_maps.shape}"
    
    average_attention_map = attention_maps.mean(0)
    average_cls_attention_map = average_attention_map[0]
    feature_importance = average_cls_attention_map[1:]
    
    assert feature_importance.shape == (n_features,)
    
    return feature_importance.numpy()


def extract_sparse_ftt_importance(model, X_num, X_cat):
    """Extrait l'importance pour Sparse FTT+ avec attention interprétable"""
    hook = SparseAttentionExtractor()
    hooks = []
    
    def patched_forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        """Forward patché pour capturer l'attention moyennée"""
        from sparsemax import Sparsemax
        sparsemax = Sparsemax(dim=-1)
        batch_size, seq_len_q, d_model = x_q.shape
        seq_len_k = x_kv.shape[1]
        
        q = self.W_q(x_q)
        k = self.W_k(x_kv)
        v = self.W_v(x_kv)
        
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_head).transpose(1, 2)
        
        attention_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs_per_head = sparsemax(attention_logits)
        
        if self.dropout is not None:
            attention_probs_per_head = self.dropout(attention_probs_per_head)
        
        avg_attention = attention_probs_per_head.mean(dim=1)
        output = torch.matmul(avg_attention, v)
        
        if self.W_h is not None:
            output = self.W_h(output)
        
        return output, avg_attention
    
    original_forwards = []
    for layer in model.layers:
        attention_module = layer['attention']
        original_forwards.append(attention_module.forward)
        attention_module.forward = patched_forward.__get__(attention_module)
        h = attention_module.register_forward_hook(hook)
        hooks.append(h)
    
    model.eval()
    with torch.no_grad():
        batch_size = 128
        n_samples = X_num.shape[0] if X_num is not None else X_cat.shape[0]
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_num_batch = X_num[i:end_idx] if X_num is not None else None
            X_cat_batch = X_cat[i:end_idx] if X_cat is not None else None
            _ = model(X_num_batch, X_cat_batch)
    
    for h in hooks:
        h.remove()
    for layer, original_forward in zip(model.layers, original_forwards):
        layer['attention'].forward = original_forward
    
    if not hook.attention_maps:
        raise ValueError("No attention maps collected")
    
    attention_maps = torch.cat(hook.attention_maps)
    n_features = (0 if X_num is None else X_num.shape[1]) + (0 if X_cat is None else X_cat.shape[1])
    n_tokens = n_features + 1
    
    average_attention_map = attention_maps.mean(0)
    average_cls_attention_map = average_attention_map[0]
    feature_importance = average_cls_attention_map[1:]
    
    assert feature_importance.shape == (n_features,)
    
    return feature_importance.numpy()


def permutation_test_importance(model, X_num, X_cat, y, loss_fn):
    """Test de permutation pour importance des caractéristiques, sans limitation d'échantillons"""
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        pred_baseline = model(X_num, X_cat)
        baseline_loss = loss_fn(pred_baseline, y).item()
    
    importance_scores = []
    n_num_features = 0 if X_num is None else X_num.shape[1]
    n_cat_features = 0 if X_cat is None else X_cat.shape[1]
    total_features = n_num_features + n_cat_features
    
    for feat_idx in tqdm(range(total_features), desc="Permutation test"):
        X_num_perm = X_num.clone() if X_num is not None else None
        X_cat_perm = X_cat.clone() if X_cat is not None else None
        
        if feat_idx < n_num_features and X_num_perm is not None:
            perm_indices = torch.randperm(X_num_perm.shape[0], device=device)
            X_num_perm[:, feat_idx] = X_num_perm[perm_indices, feat_idx]
        elif X_cat_perm is not None:
            cat_idx = feat_idx - n_num_features
            perm_indices = torch.randperm(X_cat_perm.shape[0], device=device)
            X_cat_perm[:, cat_idx] = X_cat_perm[perm_indices, cat_idx]
        
        with torch.no_grad():
            pred_perm = model(X_num_perm, X_cat_perm)
            perm_loss = loss_fn(pred_perm, y).item()
        
        importance = perm_loss - baseline_loss
        importance_scores.append(importance)
    
    importance_scores = np.array(importance_scores)
    if baseline_loss != 0:
        importance_scores /= baseline_loss
    return importance_scores


def load_test_data(dataset_name, seed, normalization=None):
    """Charge les données du test set avec une graine et normalisation spécifiques"""
    dataset_dir = Path(f"rtdl_revisiting_models/data/{dataset_name}")
    D = lib.Dataset.from_dir(dataset_dir)
    info = D.info

    result = D.build_X(
        normalization=normalization,
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy='indices',
        cat_min_frequency=0.0,
        seed=seed,
    )

    if isinstance(result, tuple):
        N_dict, C_dict = result
    else:
        N_dict, C_dict = result, None

    X_num = torch.from_numpy(N_dict['test']).float() if N_dict is not None else None
    X_cat = torch.from_numpy(C_dict['test']).long() if C_dict is not None else None
    y = torch.from_numpy(D.y['test']).float() if info['task_type'] != 'multiclass' else torch.from_numpy(D.y['test']).long()

    return X_num, X_cat, y, info


def load_trained_model(model_name, dataset_name, seed):
    """Charge un modèle entraîné depuis checkpoint et les données de test correspondantes"""
    checkpoint_path = Path(f"rtdl_revisiting_models/output/{dataset_name}/{model_name}/default/checkpoint.pt")
    config_path = Path(f"rtdl_revisiting_models/output/{dataset_name}/{model_name}/default/{seed}.toml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    args, _ = lib.load_config(config_path)
    
    # Charger les données avec la même graine et normalisation que l'entraînement
    X_num, X_cat, y, info = load_test_data(
        dataset_name,
        seed=args.get('seed', seed),
        normalization=args['data'].get('normalization')
    )
    
    if model_name == 'ft_transformer':
        from rtdl_revisiting_models.bin.ft_transformer import Transformer
    elif model_name == 'sparse_ftt_plus':
        from rtdl_revisiting_models.bin.sparse_ftt_plus import Transformer
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = Transformer(
        d_numerical=0 if X_num is None else X_num.shape[1],
        categories=lib.get_categories(X_cat) if X_cat is not None else None,
        d_out=info['n_classes'] if info['task_type'] == 'multiclass' else 1,
        **args['model'],
    )
    
    device = lib.get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    
    return model, info, X_num, X_cat, y


def analyze_dataset(dataset_name, seeds=[0, 1, 2]):
    """Analyse complète pour un dataset, retourne les corrélations pour sauvegarde"""
    print(f"\n{'='*60}")
    print(f"Analyzing dataset: {dataset_name.upper()}")
    print('='*60)
    
    models = ['ft_transformer', 'sparse_ftt_plus']
    all_correlations = {model: [] for model in models}
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        
        for model_name in models:
            print(f"  Processing {model_name}...")
            
            try:
                model, info, X_num_test, X_cat_test, y_test = load_trained_model(model_name, dataset_name, seed)
                
                device = lib.get_device()
                if X_num_test is not None:
                    X_num_test = X_num_test.to(device)
                if X_cat_test is not None:
                    X_cat_test = X_cat_test.to(device)
                y_test = y_test.to(device)
                
                if info['task_type'] == 'binclass':
                    loss_fn = F.binary_cross_entropy_with_logits
                elif info['task_type'] == 'multiclass':
                    loss_fn = F.cross_entropy
                else:
                    loss_fn = F.mse_loss
                
                if model_name == 'ft_transformer':
                    attention_importance = extract_ft_transformer_importance(model, X_num_test, X_cat_test)
                else:
                    attention_importance = extract_sparse_ftt_importance(model, X_num_test, X_cat_test)
                
                perm_importance = permutation_test_importance(
                    model, X_num_test, X_cat_test, y_test, loss_fn
                )
                
                correlation, _ = spearmanr(attention_importance, perm_importance)
                
                print(f"    Rank correlation: {correlation:.3f}")
                all_correlations[model_name].append(correlation)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    print(f"\n{dataset_name.upper()} - FINAL RESULTS:")
    print("-" * 40)
    
    for model_name in models:
        if all_correlations[model_name]:
            mean_corr = np.mean(all_correlations[model_name])
            std_corr = np.std(all_correlations[model_name])
            print(f"{model_name:>15}: {mean_corr:.2f} ({std_corr:.2f})")
    
    return all_correlations


def main():
    """Analyse principale, sauvegarde les résultats dans un JSON"""
    datasets = ['adult', 'california_housing']  # BOA_dataset sera ajouté plus tard
    seeds = [0, 1, 2]
    
    print("🔍 Feature Importance Analysis: FT-Transformer vs Sparse FTT+")
    print("Comparing attention maps with permutation test (Breiman 2001)")
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = analyze_dataset(dataset, seeds)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - Rank Correlation (Mean ± Std)")
    print('='*80)
    
    header = f"{'Model':<25}"
    for dataset in datasets:
        header += f"{dataset.upper():>20}"
    print(header)
    print("-" * len(header))
    
    json_results = {"datasets": {}}
    
    for model in ['ft_transformer', 'sparse_ftt_plus']:
        row = f"{model:<25}"
        for dataset in datasets:
            if dataset in all_results and model in all_results[dataset]:
                corrs = all_results[dataset][model]
                if corrs:
                    mean_c = np.mean(corrs)
                    std_c = np.std(corrs)
                    row += f"{mean_c:.2f} ({std_c:.2f}):>20"
                    if dataset not in json_results["datasets"]:
                        json_results["datasets"][dataset] = {}
                    json_results["datasets"][dataset][model] = {
                        "correlations": [float(c) for c in corrs],
                        "mean": float(mean_c),
                        "std": float(std_c)
                    }
                else:
                    row += f"{'N/A':>20}"
                    if dataset not in json_results["datasets"]:
                        json_results["datasets"][dataset] = {}
                    json_results["datasets"][dataset][model] = {
                        "correlations": [],
                        "mean": None,
                        "std": None
                    }
            else:
                row += f"{'N/A':>20}"
                if dataset not in json_results["datasets"]:
                    json_results["datasets"][dataset] = {}
                json_results["datasets"][dataset][model] = {
                    "correlations": [],
                    "mean": None,
                    "std": None
                }
        print(row)
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "feature_importance_results.json"
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()