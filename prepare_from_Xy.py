"""
prepare_from_Xy.py
Prépare et sauvegarde les fichiers N_*, C_*, y_* et info.json dans une structure compatible avec
rtdl_revisiting_models/data/<dataset>/* à partir de X, y numpy arrays, avec option d'oversampling.
- Les variables catégoriques sont sauvegardées en valeurs brutes (strings) pour que build_X gère l'encodage.
- Supporte un oversampler personnalisé ou MGSGRFOverSampler par défaut.

Usage:
    from prepare_from_Xy import prepare_and_save_dataset
    prepare_and_save_dataset(X, y, numeric_features, categorical_features,
                             out_dir='rtdl_revisiting_models/data/bankchurners',
                             test_size=0.2, val_size=0.2, random_state=42, oversampler=None)
"""
from __future__ import annotations
import json
import os
from typing import Sequence, Optional, Callable, Tuple
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from mgs_grf import MGSGRFOverSampler  # Importer l'oversampler par défaut

def _ensure_directory(path: str) -> None:
    """Crée un répertoire s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)

def _extract_numeric_data(X: np.ndarray, numeric_idx: Sequence[int]) -> np.ndarray:
    """Extrait les caractéristiques numériques avec dtype float32."""
    if not numeric_idx:
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    return X[:, numeric_idx].astype(np.float32)

def _extract_categorical_data(X: np.ndarray, cat_idx: Sequence[int]) -> np.ndarray:
    """Extrait les caractéristiques catégoriques en valeurs brutes (strings)."""
    if not cat_idx:
        return np.zeros((X.shape[0], 0), dtype='U')  # Utilisation de 'U' pour les chaînes Unicode
    return X[:, cat_idx].astype(str)

def prepare_and_save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    numeric_features: Sequence[int],
    categorical_features: Sequence[int],
    out_dir: str = "rtdl_revisiting_models/data/bankchurners",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    basename: Optional[str] = None,
    oversampler: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    """
    Prépare les splits train/val/test et sauvegarde les fichiers pour rtdl_revisiting_models.
    
    Args:
        X: Matrice des features (numpy array)
        y: Vecteur des labels (numpy array)
        numeric_features: Indices des features numériques
        categorical_features: Indices des features catégoriques
        out_dir: Répertoire de sortie
        test_size: Proportion des données de test
        val_size: Proportion des données de validation
        random_state: Graine pour la reproductibilité
        basename: Nom de base du dataset
        oversampler: Fonction ou objet callable (ex. MGSGRFOverSampler().fit_resample) 
                     pour oversampler train. Si None, pas d'oversampling.
    """
    # Vérifications initiales
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0], "X et y doivent avoir le même nombre de lignes"

    # Créer le répertoire de sortie
    _ensure_directory(out_dir)
    if basename is None:
        basename = os.path.basename(out_dir.rstrip('/')) or 'dataset'

    # Diviser les données en train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    # Diviser train / val
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=random_state,
        stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None
    )

    # Appliquer l'oversampling si spécifié (uniquement sur train)
    if oversampler is not None:
        X_train, y_train = oversampler(X_train, y_train)
        try:
            print("Distribution après oversampling :", Counter(y_train))
        except Exception:
            print("Impossible d'afficher la distribution après oversampling.")

    # Extraire les données numériques et catégoriques
    N_train = _extract_numeric_data(X_train, numeric_features)
    N_val = _extract_numeric_data(X_val, numeric_features)
    N_test = _extract_numeric_data(X_test, numeric_features)

    C_train = _extract_categorical_data(X_train, categorical_features)
    C_val = _extract_categorical_data(X_val, categorical_features)
    C_test = _extract_categorical_data(X_test, categorical_features)

    # Sauvegarder les tableaux (train, val, test)
    np.save(os.path.join(out_dir, 'N_train.npy'), N_train)
    np.save(os.path.join(out_dir, 'N_val.npy'), N_val)
    np.save(os.path.join(out_dir, 'N_test.npy'), N_test)

    np.save(os.path.join(out_dir, 'C_train.npy'), C_train)
    np.save(os.path.join(out_dir, 'C_val.npy'), C_val)
    np.save(os.path.join(out_dir, 'C_test.npy'), C_test)

    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)

    # Concaténer train et val pour trainval
    N_trainval = np.concatenate([N_train, N_val], axis=0)
    C_trainval = np.concatenate([C_train, C_val], axis=0) if C_train.size or C_val.size else np.zeros((N_trainval.shape[0], 0), dtype='U')
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    np.save(os.path.join(out_dir, 'N_trainval.npy'), N_trainval)
    np.save(os.path.join(out_dir, 'C_trainval.npy'), C_trainval)
    np.save(os.path.join(out_dir, 'y_trainval.npy'), y_trainval)

    # Préparer et sauvegarder info.json
    task_type = 'binclass' if len(np.unique(y)) <= 2 else 'multiclass' if y.dtype.kind in 'iu' else 'regression'
    info = {
        'name': f'{basename}___0',
        'basename': basename,
        'split': 0,
        'task_type': task_type,
        'n_num_features': len(numeric_features),
        'n_cat_features': len(categorical_features),
        'train_size': N_train.shape[0],
        'val_size': N_val.shape[0],
        'test_size': N_test.shape[0],
    }
    with open(os.path.join(out_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    # Afficher un résumé clair
    print(f"Dataset sauvegardé dans {out_dir}")
    print(f"Tailles numériques : train={N_train.shape}, val={N_val.shape}, test={N_test.shape}")
    print(f"Tailles catégoriques : train={C_train.shape}, val={C_val.shape}, test={C_test.shape}")
    print(f"Types et tailles de y : type={y_train.dtype}, train={y_train.shape}, val={y_val.shape}, test={y_test.shape}")

if __name__ == "__main__":
    # Exemple d'utilisation
    np.random.seed(42)
    numeric_features = [0, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    categorical_features = [1, 3, 4, 5, 6]
    # Initialiser X avec dtype=object pour accepter floats et strings
    X = np.empty((1000, 19), dtype=object)
    # Remplir les colonnes numériques avec des valeurs aléatoires
    X[:, numeric_features] = np.random.rand(1000, len(numeric_features))
    # Remplir les colonnes catégoriques avec des chaînes
    X[:, categorical_features] = np.random.choice(['A', 'B', 'C'], size=(1000, len(categorical_features)))
    y = np.random.randint(0, 2, 1000)  # Classification binaire

    # Cas sans oversampling
    prepare_and_save_dataset(X, y, numeric_features, categorical_features, out_dir="rtdl_revisiting_models/data/bankchurners_example")

    # Cas avec oversampling
    def mgs_oversampler(X, y):
        mgs_grf = MGSGRFOverSampler(K=len(numeric_features), categorical_features=categorical_features, random_state=0)
        return mgs_grf.fit_resample(X, y)

    prepare_and_save_dataset(
        X, y, numeric_features, categorical_features,
        out_dir="rtdl_revisiting_models/data/bankchurners_oversampled_example",
        oversampler=mgs_oversampler
    )