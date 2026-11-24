"""
Modul Pipeline Pelatihan, Validasi & Inferensi (pipeline.py)
============================================================
Pipeline Lengkap Sesuai Dokumen Konteks Sistem (Full Library):
1. Dataset Loader & Sampling:
   - Sampling 1000 gambar per kelas (total 6000 untuk 6 kelas).
   - Stratified Train/Test Split (80:20 -> 4800 train, 1200 test) PALING AWAL sebelum fitting (anti-leakage).
2. Ekstraksi Fitur:
   - Jalur Utama: MobileNetV3-Small (576 dimensi float, Keras pretrained ImageNet).
   - Jalur Pembanding: Handcrafted (~26 dimensi: 16 HSV Hist + 6 Warna + 4 GLCM).
3. StandardScaler (sklearn.preprocessing.StandardScaler):
   - HANYA di-fit pada data latih (scaler.fit(X_train)), lalu transform ke data uji/baru.
4. PCA (sklearn.decomposition.PCA):
   - HANYA di-fit pada data latih terskala (X_train_scaled).
   - Default n_components=50 (atau target explained variance ratio misal 0.95, dibatasi min(n_comp, n_samples, n_features)).
   - Menyimpan koordinat PC1 & PC2 untuk visualisasi scatter plot.
5. Fuzzy C-Means (scikit-fuzzy: skfuzzy.cluster.cmeans & cmeans_predict):
   - HANYA di-fit pada X_train_pca. Menghasilkan pusat cluster V, keanggotaan mu_train, FPC, PE, Silhouette.
   - Keanggotaan data uji mu_test dihitung melalui fcm_predict_membership (cmeans_predict) terhadap pusat cluster V.
6. Integrasi Fitur Gabungan:
   - Vektor input KNN: [PCA || alpha * mu] dengan bobot alpha (default 1.0).
7. Klasifikasi KNN (sklearn.neighbors.KNeighborsClassifier):
   - Metrik Euclidean, default k=5, opsi weights ("uniform" atau "distance").
   - Dilatih dengan label kelas asli.
8. 4 Skenario Pengujian Eksperimen (1a, 1b, 2a, 2b):
   - 1a: MobileNetV3-Small + PCA + (FCM + KNN) [Deep Learning + Hybrid FCM-KNN]
   - 1b: MobileNetV3-Small + PCA + KNN [Deep Learning Baseline]
   - 2a: (Histogram HSV + GLCM) + PCA + (FCM + KNN) [Handcrafted + Hybrid FCM-KNN]
   - 2b: (Histogram HSV + GLCM) + PCA + KNN [Handcrafted Baseline]
9. Helper Hyperparameter Tuning k in [3, 5, 7, 9], weights, alpha via Cross-Validation pada data latih.
10. Evaluasi Lengkap: Akurasi, Presisi, Recall, F1 (per kelas & macro/weighted), Confusion Matrix.
11. Penyimpanan & Pemuatan Model (save_models / load_models) 5 artefak:
    scaler.pkl, pca.pkl, fcm_centers.npy, knn.pkl, meta.json.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

# Coba import joblib, fallback ke pickle
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle as joblib
    JOBLIB_AVAILABLE = False

# Import modul internal
from .fcm import (
    FCM,
    fcm_predict_membership,
    compute_fpc,
    compute_partition_entropy,
    compute_fcm_silhouette,
)
from .features import (
    extract_features,
    extract_mobilenet_features,
    extract_handcrafted_features,
    get_feature_names,
)

# Coba import scikit-learn
try:
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    from sklearn.model_selection import (
        train_test_split as sklearn_train_test_split,
        StratifiedKFold as SklearnStratifiedKFold,
    )
    from sklearn.metrics import (
        accuracy_score as sklearn_accuracy_score,
        precision_score as sklearn_precision_score,
        recall_score as sklearn_recall_score,
        f1_score as sklearn_f1_score,
        confusion_matrix as sklearn_confusion_matrix,
        classification_report as sklearn_classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# Helper Serialisasi JSON untuk Tipe Data NumPy
# ==============================================================================

def _to_json_serializable(obj: Any) -> Any:
    """Konversi rekursif tipe data NumPy ke tipe Python murni agar valid di JSON."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(item) for item in obj]
    return obj


# ==============================================================================
# Fallback Scikit-Learn (Jika Belum Terinstal di Environment)
# ==============================================================================

class _NumpyStandardScaler:
    """Implementasi fallback StandardScaler berbasis NumPy."""
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "_NumpyStandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class _NumpyPCA:
    """Implementasi fallback PCA berbasis SVD NumPy."""
    def __init__(self, n_components: Union[int, float] = 50, random_state: Optional[int] = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Any = None) -> "_NumpyPCA":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD: X = U * S * Vt
        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        total_var = np.sum(s ** 2) / (n_samples - 1 if n_samples > 1 else 1)

        exp_var_all = (s ** 2) / (n_samples - 1 if n_samples > 1 else 1)
        exp_ratio_all = exp_var_all / (total_var if total_var > 0 else 1.0)

        if isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            cum_var = np.cumsum(exp_ratio_all)
            idx_reached = np.where(cum_var >= self.n_components)[0]
            k = idx_reached[0] + 1 if len(idx_reached) > 0 else len(exp_var_all)
        else:
            k = int(self.n_components)

        k = max(1, min(k, n_samples, n_features))
        self.components_ = vt[:k, :]
        self.explained_variance_ = exp_var_all[:k]
        self.explained_variance_ratio_ = exp_ratio_all[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X - self.mean_, self.components_.T)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class _NumpyKNN:
    """Implementasi fallback KNeighborsClassifier berbasis NumPy dengan dukungan metric euclidean & cosine."""
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = str(metric).lower()
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyKNN":
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            norm_X = np.linalg.norm(X, axis=1, keepdims=True)
            norm_tr = np.linalg.norm(self.X_train_, axis=1, keepdims=True)
            norm_X = np.where(norm_X < 1e-12, 1.0, norm_X)
            norm_tr = np.where(norm_tr < 1e-12, 1.0, norm_tr)
            X_normed = X / norm_X
            tr_normed = self.X_train_ / norm_tr
            cosine_sim = np.dot(X_normed, tr_normed.T)
            return np.clip(1.0 - cosine_sim, 0.0, 2.0)
        else:
            diff = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
            return np.linalg.norm(diff, axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("KNN belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dists = self._compute_distances(X)

        k = max(1, min(self.n_neighbors, self.X_train_.shape[0]))
        nn_indices = np.argsort(dists, axis=1)[:, :k]

        preds = []
        for i, row in enumerate(nn_indices):
            neighbor_labels = self.y_train_[row]
            if self.weights == "distance":
                row_dists = dists[i, row]
                weights = np.where(row_dists < 1e-12, 1e6, 1.0 / (row_dists + 1e-12))
                unique_labels = np.unique(neighbor_labels)
                label_scores = {lbl: np.sum(weights[neighbor_labels == lbl]) for lbl in unique_labels}
                best_label = max(label_scores.items(), key=lambda item: item[1])[0]
                preds.append(best_label)
            else:
                vals, counts = np.unique(neighbor_labels, return_counts=True)
                preds.append(vals[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None or self.classes_ is None:
            raise RuntimeError("KNN belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        dists = self._compute_distances(X)

        k = max(1, min(self.n_neighbors, self.X_train_.shape[0]))
        nn_indices = np.argsort(dists, axis=1)[:, :k]

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for i, row in enumerate(nn_indices):
            neighbor_labels = self.y_train_[row]
            if self.weights == "distance":
                row_dists = dists[i, row]
                weights = np.where(row_dists < 1e-12, 1e6, 1.0 / (row_dists + 1e-12))
                for lbl, w in zip(neighbor_labels, weights):
                    if lbl in class_to_idx:
                        probas[i, class_to_idx[lbl]] += w
                total_w = np.sum(probas[i])
                if total_w > 0:
                    probas[i] /= total_w
            else:
                for lbl in neighbor_labels:
                    if lbl in class_to_idx:
                        probas[i, class_to_idx[lbl]] += 1.0 / k
        return probas


def _get_standard_scaler():
    if SKLEARN_AVAILABLE:
        return SklearnStandardScaler()
    return _NumpyStandardScaler()


def _get_pca(n_components: Union[int, float] = 50, random_state: Optional[int] = 42):
    if SKLEARN_AVAILABLE:
        return SklearnPCA(n_components=n_components, random_state=random_state)
    return _NumpyPCA(n_components=n_components, random_state=random_state)


def _get_knn(n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"):
    if SKLEARN_AVAILABLE:
        return SklearnKNN(n_neighbors=n_neighbors, weights=weights, metric=metric)
    return _NumpyKNN(n_neighbors=n_neighbors, weights=weights, metric=metric)


def _stratified_split(
    X_items: List[Any],
    y_items: Union[np.ndarray, List[Any]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Any], List[Any], np.ndarray, np.ndarray]:
    """Stratified train/test split deterministik untuk mencegah data leakage."""
    if SKLEARN_AVAILABLE:
        try:
            return sklearn_train_test_split(
                X_items, y_items, test_size=test_size, stratify=y_items, random_state=random_state
            )
        except Exception:
            pass

    # Fallback NumPy
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y_items)
    unique_classes = np.unique(y_arr)

    train_indices = []
    test_indices = []

    for c in unique_classes:
        c_indices = np.where(y_arr == c)[0]
        rng.shuffle(c_indices)
        n_test = max(1, int(round(len(c_indices) * test_size)))
        test_indices.extend(c_indices[:n_test])
        train_indices.extend(c_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    X_train = [X_items[i] for i in train_indices] if isinstance(X_items, list) else X_items[train_indices]
    X_test = [X_items[i] for i in test_indices] if isinstance(X_items, list) else X_items[test_indices]
    y_train = y_arr[train_indices]
    y_test = y_arr[test_indices]

    return X_train, X_test, y_train, y_test


# ==============================================================================
# Evaluasi Metrik Klasifikasi Supervised
# ==============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi metrik klasifikasi scikit-learn:
    - Accuracy
    - Macro Precision, Macro Recall, Macro F1
    - Weighted Precision, Weighted Recall, Weighted F1
    - Per-class Precision, Recall, F1
    - Confusion Matrix
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if labels is None:
        all_labels = np.unique(np.concatenate([y_t, y_p]))
    else:
        all_labels = np.asarray(labels)

    if SKLEARN_AVAILABLE:
        try:
            acc = float(sklearn_accuracy_score(y_t, y_p))
            prec_macro = float(sklearn_precision_score(y_t, y_p, average="macro", zero_division=0))
            rec_macro = float(sklearn_recall_score(y_t, y_p, average="macro", zero_division=0))
            f1_macro = float(sklearn_f1_score(y_t, y_p, average="macro", zero_division=0))

            prec_weighted = float(sklearn_precision_score(y_t, y_p, average="weighted", zero_division=0))
            rec_weighted = float(sklearn_recall_score(y_t, y_p, average="weighted", zero_division=0))
            f1_weighted = float(sklearn_f1_score(y_t, y_p, average="weighted", zero_division=0))

            prec_per_class = sklearn_precision_score(y_t, y_p, labels=all_labels, average=None, zero_division=0).tolist()
            rec_per_class = sklearn_recall_score(y_t, y_p, labels=all_labels, average=None, zero_division=0).tolist()
            f1_per_class = sklearn_f1_score(y_t, y_p, labels=all_labels, average=None, zero_division=0).tolist()

            cm = sklearn_confusion_matrix(y_t, y_p, labels=all_labels).tolist()

            per_class_metrics = {}
            for i, lbl in enumerate(all_labels):
                per_class_metrics[str(lbl)] = {
                    "precision": float(prec_per_class[i]),
                    "recall": float(rec_per_class[i]),
                    "f1_score": float(f1_per_class[i]),
                }

            return {
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "precision_weighted": prec_weighted,
                "recall_weighted": rec_weighted,
                "f1_weighted": f1_weighted,
                "per_class": per_class_metrics,
                "confusion_matrix": cm,
                "labels": all_labels.tolist(),
            }
        except Exception:
            pass

    # Fallback NumPy murni
    acc = float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

    precisions, recalls, f1s, supports = [], [], [], []
    per_class_metrics = {}
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)

    for t, p in zip(y_t, y_p):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    for lbl in all_labels:
        tp = np.sum((y_t == lbl) & (y_p == lbl))
        fp = np.sum((y_t != lbl) & (y_p == lbl))
        fn = np.sum((y_t == lbl) & (y_p != lbl))
        sup = np.sum(y_t == lbl)

        p_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0

        precisions.append(p_c)
        recalls.append(r_c)
        f1s.append(f_c)
        supports.append(sup)

        per_class_metrics[str(lbl)] = {
            "precision": float(p_c),
            "recall": float(r_c),
            "f1_score": float(f_c),
        }

    total_sup = sum(supports) if sum(supports) > 0 else 1
    weighted_p = float(sum(p * s for p, s in zip(precisions, supports)) / total_sup)
    weighted_r = float(sum(r * s for r, s in zip(recalls, supports)) / total_sup)
    weighted_f = float(sum(f * s for f, s in zip(f1s, supports)) / total_sup)

    return {
        "accuracy": acc,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels.tolist(),
    }


# ==============================================================================
# Pemuatan Dataset Dinamis dari Direktori
# ==============================================================================

def load_dataset_from_directory(
    data_dir: Union[str, Path],
    samples_per_class: int = 1000,
    random_state: int = 42,
    deduplicate_pixels: bool = False,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Memuat dataset citra dari subfolder kelas secara dinamis (tanpa hardcode nama kelas).
    Default mengambil hingga 1000 citra per kelas.
    
    Jika `deduplicate_pixels=True`, memverifikasi SHA-256 piksel citra agar tidak ada
    citra identik/duplikat di dalam dataset (sesuai Best Result / anti-leakage tingkat lanjut).

    Parameters
    ----------
    data_dir : str atau Path
        Direktori induk berisi subfolder kelas (misal: data/Garbage_Dataset_Classification/images).
    samples_per_class : int, default=1000
        Batas maksimum sampel yang diambil per kelas.
    random_state : int, default=42
    deduplicate_pixels : bool, default=False
        Opsi filter hash piksel citra duplikat.

    Returns
    -------
    tuple of (filepaths, labels, class_names)
    """
    import hashlib
    from PIL import Image

    base_path = Path(data_dir)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {data_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rng = np.random.RandomState(random_state)

    filepaths: List[str] = []
    labels: List[str] = []
    class_names: List[str] = []
    seen_hashes = set()

    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"Tidak ada subfolder kelas ditemukan di {data_dir}")

    for sub in subdirs:
        c_name = sub.name
        class_names.append(c_name)

        img_files = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        img_files = sorted(img_files, key=lambda x: x.name)

        if len(img_files) == 0:
            continue

        if deduplicate_pixels:
            unique_files = []
            for f in img_files:
                try:
                    with Image.open(f) as img:
                        digest = hashlib.sha256(f"{img.width}x{img.height}:RGB:".encode())
                        digest.update(img.convert("RGB").tobytes())
                        h = digest.hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_files.append(f)
                except Exception:
                    unique_files.append(f)
            img_files = unique_files

        if len(img_files) > samples_per_class:
            chosen = rng.choice(img_files, size=samples_per_class, replace=False)
        else:
            chosen = img_files

        for f in chosen:
            filepaths.append(str(f))
            labels.append(c_name)

    return filepaths, np.array(labels), class_names


def load_default_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi dari config.yaml atau mengembalikan dictionary default jika file tidak ada."""
    p = Path(config_path)
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            pass

    return {
        "dataset": {
            "dataset_slug": "zlatan599/garbage-dataset-classification",
            "data_dir": "data/Garbage_Dataset_Classification/images",
            "samples_per_class": 1000,
            "test_size": 0.2,
            "random_state": 42,
        },
        "features": {
            "feature_extractor": "mobilenetv3_small",
            "n_features": 576,
            "image_size": [224, 224],
            "use_edge_density": False,
        },
        "pca": {
            "n_components": 50,
            "variance_threshold": 0.95,
        },
        "fcm": {
            "c": 6,
            "m": 2.0,
            "maxiter": 200,
            "error": 0.00001,
            "metric": "euclidean",
            "random_state": 42,
        },
        "knn": {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "cosine",
            "alpha": 1.0,
            "mu_weight": 1.0,
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
        },
    }


# ==============================================================================
# Helper Hyperparameter Tuning KNN (k, weights, alpha)
# ==============================================================================

def tune_knn_hyperparameters(
    X_train_pca: np.ndarray,
    mu_train: np.ndarray,
    y_train: np.ndarray,
    k_values: List[int] = [3, 5, 7, 9],
    weights_options: List[str] = ["uniform", "distance"],
    metric_options: List[str] = ["euclidean", "cosine"],
    alpha_values: List[float] = [0.5, 1.0, 1.5, 2.0],
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Melakukan validasi silang (cross-validation) pada data latih untuk mencari kombinasi
    parameter k terbaik [3, 5, 7, 9], opsi weights ('uniform' vs 'distance'), metric ('euclidean' vs 'cosine'),
    dan alpha scaling.
    """
    y_arr = np.asarray(y_train)
    n_samples = len(y_arr)
    cv_actual = max(2, min(cv, n_samples // len(np.unique(y_arr)) if len(np.unique(y_arr)) > 0 else 2))

    # Bagi fold dengan StratifiedKFold
    folds = []
    if SKLEARN_AVAILABLE:
        try:
            skf = SklearnStratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=random_state)
            for tr_idx, val_idx in skf.split(X_train_pca, y_arr):
                folds.append((tr_idx, val_idx))
        except Exception:
            pass

    if not folds:
        # Fallback fold generation
        rng = np.random.RandomState(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        fold_sizes = np.full(cv_actual, n_samples // cv_actual, dtype=int)
        fold_sizes[: n_samples % cv_actual] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            tr_idx = np.setdiff1d(indices, val_idx)
            folds.append((tr_idx, val_idx))
            current = stop

    cv_results = []
    best_score = -1.0
    best_params = {}

    for alpha in alpha_values:
        X_train_full = np.hstack([X_train_pca, mu_train * float(alpha)])
        for met in metric_options:
            for w in weights_options:
                for k in k_values:
                    fold_scores = []
                    for tr_idx, val_idx in folds:
                        X_tr_fold = X_train_full[tr_idx]
                        y_tr_fold = y_arr[tr_idx]
                        X_val_fold = X_train_full[val_idx]
                        y_val_fold = y_arr[val_idx]

                        k_clamped = max(1, min(k, len(X_tr_fold) - 1))
                        knn_model = _get_knn(n_neighbors=k_clamped, weights=w, metric=met)
                        knn_model.fit(X_tr_fold, y_tr_fold)
                        val_preds = knn_model.predict(X_val_fold)
                        acc = float(np.mean(val_preds == y_val_fold))
                        fold_scores.append(acc)

                    mean_acc = float(np.mean(fold_scores))
                    std_acc = float(np.std(fold_scores))
                    record = {
                        "k": k,
                        "weights": w,
                        "metric": met,
                        "alpha": alpha,
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                    }
                    cv_results.append(record)

                    if mean_acc > best_score:
                        best_score = mean_acc
                        best_params = {"k": k, "weights": w, "metric": met, "alpha": alpha, "cv_score": best_score}

    return {
        "best_params": best_params,
        "cv_results": cv_results,
    }


# ==============================================================================
# Helper Evaluasi 4 Skenario Pengujian (1a, 1b, 2a, 2b)
# ==============================================================================

def run_four_scenarios(
    X_train_raw: List[Any],
    y_train: np.ndarray,
    X_test_raw: Optional[List[Any]] = None,
    y_test: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Menjalankan dan mengevaluasi 4 skenario eksperimen pada data train/test split yang sama persis:
    - Skenario 1a: MobileNetV3-Small + PCA + (FCM + KNN) [Deep Learning + Hybrid FCM-KNN]
    - Skenario 1b: MobileNetV3-Small + PCA + KNN [Deep Learning Baseline]
    - Skenario 2a: (Histogram HSV + GLCM) + PCA + (FCM + KNN) [Handcrafted + Hybrid FCM-KNN]
    - Skenario 2b: (Histogram HSV + GLCM) + PCA + KNN [Handcrafted Baseline]
    """
    if config is None:
        config = load_default_config()

    if X_test_raw is None or y_test is None:
        test_size = float(config.get("dataset", {}).get("test_size", 0.2))
        random_state = int(config.get("dataset", {}).get("random_state", 42))
        X_train_raw, X_test_raw, y_train, y_test = _stratified_split(
            X_train_raw, y_train, test_size=test_size, random_state=random_state
        )

    if class_names is None:
        class_names = sorted(list(np.unique(np.concatenate([y_train, y_test]))))

    k = int(config.get("knn", {}).get("n_neighbors", 5))
    weights = str(config.get("knn", {}).get("weights", "uniform"))
    alpha = float(config.get("knn", {}).get("alpha", config.get("knn", {}).get("mu_weight", 1.0)))
    pca_n_comp = config.get("pca", {}).get("n_components", 50)
    var_thresh = float(config.get("pca", {}).get("variance_threshold", 0.95))
    fcm_c = int(config.get("fcm", {}).get("c", 6))
    fcm_m = float(config.get("fcm", {}).get("m", 2.0))
    fcm_maxiter = int(config.get("fcm", {}).get("maxiter", 200))
    fcm_err = float(config.get("fcm", {}).get("error", 1e-5))
    seed = int(config.get("dataset", {}).get("random_state", 42))

    scenarios_output: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # JALUR 1: MobileNetV3-Small
    # --------------------------------------------------------------------------
    try:
        # Ekstraksi fitur MobileNetV3-Small
        feat_tr_mob = np.array([
            extract_features(p, feature_extractor="mobilenetv3_small", config=config)
            for p in X_train_raw
        ], dtype=np.float64)
        feat_te_mob = np.array([
            extract_features(p, feature_extractor="mobilenetv3_small", config=config)
            for p in X_test_raw
        ], dtype=np.float64)

        # StandardScaler (fit train only)
        scaler_mob = _get_standard_scaler()
        X_tr_mob_sc = scaler_mob.fit_transform(feat_tr_mob)
        X_te_mob_sc = scaler_mob.transform(feat_te_mob)

        # PCA (fit train only)
        max_c = min(int(pca_n_comp) if isinstance(pca_n_comp, int) else 50, X_tr_mob_sc.shape[0] - 1, X_tr_mob_sc.shape[1])
        pca_mob = _get_pca(n_components=max_c, random_state=seed)
        X_tr_mob_pca = pca_mob.fit_transform(X_tr_mob_sc)
        X_te_mob_pca = pca_mob.transform(X_te_mob_sc)

        # FCM (fit train only)
        fcm_mob = FCM(n_clusters=min(fcm_c, len(X_tr_mob_pca)), m=fcm_m, max_iter=fcm_maxiter, error=fcm_err, random_state=seed)
        fcm_mob.fit(X_tr_mob_pca)
        mu_tr_mob = fcm_mob.u_
        mu_te_mob = fcm_predict_membership(X_te_mob_pca, fcm_mob.centers_, m=fcm_m, error=fcm_err, maxiter=fcm_maxiter, seed=seed)

        knn_metric = str(config.get("knn", {}).get("metric", "euclidean"))

        # Skenario 1b: MobileNetV3-Small + PCA + KNN
        knn_1b = _get_knn(n_neighbors=k, weights=weights, metric=knn_metric)
        knn_1b.fit(X_tr_mob_pca, y_train)
        preds_1b = knn_1b.predict(X_te_mob_pca)
        metrics_1b = compute_classification_metrics(y_test, preds_1b, labels=class_names)
        scenarios_output["1b"] = {
            "name": "Skenario 1b: MobileNetV3-Small + PCA + KNN",
            "feature_extractor": "mobilenetv3_small",
            "model_type": "PCA + KNN (Baseline)",
            "metrics": metrics_1b,
            "pca_components": X_tr_mob_pca.shape[1],
        }

        # Skenario 1a: MobileNetV3-Small + PCA + (FCM + KNN)
        X_tr_1a = np.hstack([X_tr_mob_pca, mu_tr_mob * alpha])
        X_te_1a = np.hstack([X_te_mob_pca, mu_te_mob * alpha])
        knn_1a = _get_knn(n_neighbors=k, weights=weights, metric=knn_metric)
        knn_1a.fit(X_tr_1a, y_train)
        preds_1a = knn_1a.predict(X_te_1a)
        metrics_1a = compute_classification_metrics(y_test, preds_1a, labels=class_names)
        scenarios_output["1a"] = {
            "name": "Skenario 1a: MobileNetV3-Small + PCA + (FCM + KNN)",
            "feature_extractor": "mobilenetv3_small",
            "model_type": "PCA + FCM + KNN (Hybrid)",
            "metrics": metrics_1a,
            "pca_components": X_tr_mob_pca.shape[1],
            "fcm_metrics": {"fpc": fcm_mob.fpc_, "pe": fcm_mob.pe_, "silhouette": fcm_mob.silhouette_},
        }
    except Exception as e:
        scenarios_output["1a"] = {"error": str(e)}
        scenarios_output["1b"] = {"error": str(e)}

    # --------------------------------------------------------------------------
    # JALUR 2: Handcrafted (Histogram HSV + GLCM)
    # --------------------------------------------------------------------------
    try:
        feat_tr_hand = np.array([
            extract_features(p, feature_extractor="handcrafted", config=config)
            for p in X_train_raw
        ], dtype=np.float64)
        feat_te_hand = np.array([
            extract_features(p, feature_extractor="handcrafted", config=config)
            for p in X_test_raw
        ], dtype=np.float64)

        scaler_hand = _get_standard_scaler()
        X_tr_hand_sc = scaler_hand.fit_transform(feat_tr_hand)
        X_te_hand_sc = scaler_hand.transform(feat_te_hand)

        max_c_hand = min(int(pca_n_comp) if isinstance(pca_n_comp, int) else 50, X_tr_hand_sc.shape[0] - 1, X_tr_hand_sc.shape[1])
        pca_hand = _get_pca(n_components=max_c_hand, random_state=seed)
        X_tr_hand_pca = pca_hand.fit_transform(X_tr_hand_sc)
        X_te_hand_pca = pca_hand.transform(X_te_hand_sc)

        fcm_hand = FCM(n_clusters=min(fcm_c, len(X_tr_hand_pca)), m=fcm_m, max_iter=fcm_maxiter, error=fcm_err, random_state=seed)
        fcm_hand.fit(X_tr_hand_pca)
        mu_tr_hand = fcm_hand.u_
        mu_te_hand = fcm_predict_membership(X_te_hand_pca, fcm_hand.centers_, m=fcm_m, error=fcm_err, maxiter=fcm_maxiter, seed=seed)

        knn_metric = str(config.get("knn", {}).get("metric", "euclidean"))

        # Skenario 2b: Handcrafted + PCA + KNN
        knn_2b = _get_knn(n_neighbors=k, weights=weights, metric=knn_metric)
        knn_2b.fit(X_tr_hand_pca, y_train)
        preds_2b = knn_2b.predict(X_te_hand_pca)
        metrics_2b = compute_classification_metrics(y_test, preds_2b, labels=class_names)
        scenarios_output["2b"] = {
            "name": "Skenario 2b: (HSV + GLCM) + PCA + KNN",
            "feature_extractor": "handcrafted",
            "model_type": "PCA + KNN (Baseline)",
            "metrics": metrics_2b,
            "pca_components": X_tr_hand_pca.shape[1],
        }

        # Skenario 2a: Handcrafted + PCA + (FCM + KNN)
        X_tr_2a = np.hstack([X_tr_hand_pca, mu_tr_hand * alpha])
        X_te_2a = np.hstack([X_te_hand_pca, mu_te_hand * alpha])
        knn_2a = _get_knn(n_neighbors=k, weights=weights, metric=knn_metric)
        knn_2a.fit(X_tr_2a, y_train)
        preds_2a = knn_2a.predict(X_te_2a)
        metrics_2a = compute_classification_metrics(y_test, preds_2a, labels=class_names)
        scenarios_output["2a"] = {
            "name": "Skenario 2a: (HSV + GLCM) + PCA + (FCM + KNN)",
            "feature_extractor": "handcrafted",
            "model_type": "PCA + FCM + KNN (Hybrid)",
            "metrics": metrics_2a,
            "pca_components": X_tr_hand_pca.shape[1],
            "fcm_metrics": {"fpc": fcm_hand.fpc_, "pe": fcm_hand.pe_, "silhouette": fcm_hand.silhouette_},
        }
    except Exception as e:
        scenarios_output["2a"] = {"error": str(e)}
        scenarios_output["2b"] = {"error": str(e)}

    return scenarios_output


# ==============================================================================
# Pipeline Training Lengkap (fit_pipeline)
# ==============================================================================

def fit_pipeline(
    X: Union[List[str], np.ndarray],
    y: Optional[Union[List[str], np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    feature_extractor: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Melatih seluruh pipeline sesuai arsitektur sistem:
    1. Stratified train/test split (80:20) paling awal SEBELUM fitting (anti-leakage).
    2. Ekstraksi fitur (MobileNetV3-Small 576D atau Handcrafted ~26D).
    3. StandardScaler HANYA di-fit pada data latih.
    4. PCA HANYA di-fit pada data latih (default n_components=50, dibatasi batas aman).
    5. FCM scikit-fuzzy HANYA di-fit pada X_train_pca -> pusat V, mu_train, FPC, PE, Silhouette.
    6. Prediksi mu_test melalui fcm_predict_membership terhadap pusat cluster V.
    7. Bentuk fitur gabungan [PCA || alpha * mu].
    8. Latih KNN (KNeighborsClassifier) pada fitur gabungan dengan label kelas asli.
    9. Evaluasi performa pada data uji dan hitung 4 skenario komparasi.

    Returns
    -------
    dict
        Objek model terlatih, data proyeksi PCA, metrik evaluasi train/test, dan komparasi skenario.
    """
    if config is None:
        config = load_default_config()

    if feature_extractor is None:
        feature_extractor = config.get("features", {}).get("feature_extractor", "mobilenetv3_small")

    dataset_cfg = config.get("dataset", {})
    test_size = float(dataset_cfg.get("test_size", 0.2))
    random_state = int(dataset_cfg.get("random_state", 42))

    pca_cfg = config.get("pca", {})
    req_n_components = pca_cfg.get("n_components", 50)
    var_thresh = float(pca_cfg.get("variance_threshold", 0.95))

    fcm_cfg = config.get("fcm", {})
    c = int(fcm_cfg.get("c", 6))
    m = float(fcm_cfg.get("m", 2.0))
    maxiter = int(fcm_cfg.get("maxiter", 200))
    error = float(fcm_cfg.get("error", 1e-5))

    knn_cfg = config.get("knn", {})
    k_neighbors = int(knn_cfg.get("n_neighbors", 5))
    weights = str(knn_cfg.get("weights", "uniform"))
    metric = str(knn_cfg.get("metric", "euclidean"))
    alpha = float(knn_cfg.get("alpha", knn_cfg.get("mu_weight", 1.0)))

    if y is None:
        if isinstance(X, np.ndarray):
            y_arr = np.zeros(len(X), dtype=int)
        else:
            raise ValueError("Label y wajib disediakan untuk alur supervised KNN dan stratified split.")
    else:
        y_arr = np.asarray(y)

    # 1. Stratified Train/Test Split (80:20) paling awal
    X_train_raw, X_test_raw, y_train, y_test = _stratified_split(
        X, y_arr, test_size=test_size, random_state=random_state
    )

    class_names = sorted(list(np.unique(y_arr)))

    # 2. Ekstraksi Fitur
    def _is_image_sample(sample: Any) -> bool:
        from PIL import Image
        if isinstance(sample, (str, Path, Image.Image)):
            return True
        if isinstance(sample, np.ndarray) and sample.ndim >= 2:
            return True
        return False

    if (isinstance(X_train_raw, list) and len(X_train_raw) > 0 and _is_image_sample(X_train_raw[0])) or \
       (isinstance(X_train_raw, np.ndarray) and X_train_raw.ndim >= 3):
        X_train_feat = np.array([
            extract_features(p, feature_extractor=feature_extractor, config=config)
            for p in X_train_raw
        ], dtype=np.float64)

        X_test_feat = np.array([
            extract_features(p, feature_extractor=feature_extractor, config=config)
            for p in X_test_raw
        ], dtype=np.float64)
    else:
        X_train_feat = np.asarray(X_train_raw, dtype=np.float64)
        X_test_feat = np.asarray(X_test_raw, dtype=np.float64)

    # 3. StandardScaler HANYA di-fit pada data latih
    scaler = _get_standard_scaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # 4. PCA HANYA di-fit pada data latih yang sudah diskalakan
    n_samples_train, n_features_raw = X_train_scaled.shape
    max_safe_comp = min(
        int(req_n_components) if isinstance(req_n_components, int) else 50,
        n_samples_train - 1 if n_samples_train > 1 else 1,
        n_features_raw
    )
    max_safe_comp = max(2, max_safe_comp)

    pca = _get_pca(n_components=max_safe_comp, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 5. FCM scikit-fuzzy HANYA di-fit pada X_train_pca
    fcm_c = min(c, len(X_train_pca))
    fcm = FCM(
        n_clusters=fcm_c,
        m=m,
        max_iter=maxiter,
        error=error,
        random_state=random_state,
    )
    fcm.fit(X_train_pca)

    fcm_centers = fcm.centers_
    mu_train = fcm.u_
    fpc_train = fcm.fpc_
    pe_train = fcm.pe_
    fcm_labels_train = fcm.labels_
    silhouette_train = fcm.silhouette_

    # 6. Prediksi keanggotaan mu_test pada data uji via fcm_predict_membership
    mu_test = fcm_predict_membership(X_test_pca, fcm_centers, m=m, error=error, maxiter=maxiter, seed=random_state)
    fpc_test = compute_fpc(mu_test)
    pe_test = compute_partition_entropy(mu_test)

    # 7. Integrasi Fitur Gabungan: [PCA || alpha * mu]
    X_train_knn = np.hstack([X_train_pca, mu_train * alpha])
    X_test_knn = np.hstack([X_test_pca, mu_test * alpha])

    # 8. Latih KNN dengan label kelas asli
    k_actual = max(1, min(k_neighbors, len(X_train_knn) - 1 if len(X_train_knn) > 1 else 1))
    knn = _get_knn(n_neighbors=k_actual, weights=weights, metric=metric)
    knn.fit(X_train_knn, y_train)

    # Prediksi KNN pada data uji
    y_pred_test = knn.predict(X_test_knn)
    knn_test_metrics = compute_classification_metrics(y_test, y_pred_test, labels=class_names)

    # 9. Jalankan 4 Skenario Pengujian Eksperimen jika masukan berupa citra/path
    scenarios = {}
    if (isinstance(X_train_raw, list) and len(X_train_raw) > 0 and _is_image_sample(X_train_raw[0])) or \
       (isinstance(X_train_raw, np.ndarray) and X_train_raw.ndim >= 3):
        try:
            scenarios = run_four_scenarios(
                X_train_raw=X_train_raw,
                y_train=y_train,
                X_test_raw=X_test_raw,
                y_test=y_test,
                config=config,
                class_names=class_names,
            )
        except Exception:
            pass

    result = {
        "scaler": scaler,
        "pca": pca,
        "fcm": fcm,
        "knn": knn,
        "fcm_centers": fcm_centers,
        "cluster_labels_train": fcm_labels_train,
        "classes": class_names,
        "class_names": class_names,
        "config": config,
        "meta": {
            "class_names": class_names,
            "feature_extractor": feature_extractor,
            "alpha": alpha,
            "mu_weight": alpha,
            "n_components": int(X_train_pca.shape[1]),
            "variance_threshold": var_thresh,
            "c": fcm_c,
            "m": m,
            "k": k_actual,
            "weights": weights,
            "timestamp": datetime.now().isoformat(),
        },
        "data": {
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "mu_train": mu_train,
            "mu_test": mu_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "pc1_pc2_train": X_train_pca[:, :2],
            "pc1_pc2_test": X_test_pca[:, :2],
        },
        "metrics": {
            "fcm_train": {
                "fpc": fpc_train,
                "pe": pe_train,
                "silhouette": silhouette_train,
                "n_iter": fcm.n_iter_,
                "final_objective": fcm.final_objective_,
                "objective_history": fcm.objective_history_,
            },
            "fcm_test": {
                "fpc": fpc_test,
                "pe": pe_test,
            },
            "pca": {
                "n_components": int(X_train_pca.shape[1]),
                "explained_variance_ratio": (
                    pca.explained_variance_ratio_.tolist()
                    if hasattr(pca, "explained_variance_ratio_") and pca.explained_variance_ratio_ is not None
                    else []
                ),
                "cumulative_variance_ratio": (
                    np.cumsum(pca.explained_variance_ratio_).tolist()
                    if hasattr(pca, "explained_variance_ratio_") and pca.explained_variance_ratio_ is not None
                    else []
                ),
                "total_variance_explained": (
                    float(np.sum(pca.explained_variance_ratio_))
                    if hasattr(pca, "explained_variance_ratio_") and pca.explained_variance_ratio_ is not None
                    else 0.0
                ),
            },
            "knn_test": knn_test_metrics,
            "scenarios": scenarios,
        },
    }

    return result


# ==============================================================================
# Inferensi Citra Tunggal (Predict)
# ==============================================================================

def transform(
    images_or_features: Union[np.ndarray, List[Any], str, Path],
    scaler: Any = None,
    pca: Any = None,
    config: Optional[Dict[str, Any]] = None,
    feature_extractor: str = "mobilenetv3_small",
) -> np.ndarray:
    """
    Mentransformasikan citra atau fitur ke dalam ruang PCA.
    """
    if scaler is None or pca is None:
        models = load_models()
        if models is None:
            raise RuntimeError("Model scaler dan pca belum ditemukan di folder models/.")
        scaler = models["scaler"]
        pca = models["pca"]

    if isinstance(images_or_features, (str, Path)) or not isinstance(images_or_features, np.ndarray):
        if not isinstance(images_or_features, list):
            images_or_features = [images_or_features]
        feats = np.array([
            extract_features(img, feature_extractor=feature_extractor, config=config)
            for img in images_or_features
        ], dtype=np.float64)
    else:
        feats = images_or_features

    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    feats_scaled = scaler.transform(feats)
    feats_pca = pca.transform(feats_scaled)
    return feats_pca


def predict(
    image_or_path: Union[str, Path, Any],
    models: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Memprediksi citra baru menggunakan model yang sudah dilatih:
    1. Ekstraksi fitur (MobileNetV3-Small / Handcrafted).
    2. Transformasi StandardScaler.
    3. Proyeksi ke ruang PCA.
    4. Hitung derajat keanggotaan mu_ij via fcm_predict_membership terhadap fcm_centers.
    5. Bentuk fitur input KNN: [PCA || alpha * mu].
    6. Prediksi kelas dan probabilitas dengan KNN.
    7. Teks interpretasi ramah bahasa Indonesia.
    """
    if models is None:
        models = load_models()
        if models is None:
            raise RuntimeError(
                "Model belum dilatih atau file di folder models/ belum lengkap. "
                "Silakan jalankan notebook di Google Colab atau latih ulang via menu 'Proses'."
            )

    scaler = models["scaler"]
    pca = models["pca"]
    fcm_centers = models["fcm_centers"]
    knn = models["knn"]
    meta = models.get("meta", {})
    class_names = models.get("class_names", meta.get("class_names", []))

    if config is None:
        config = models.get("config", load_default_config())

    feature_extractor = meta.get("feature_extractor", config.get("features", {}).get("feature_extractor", "mobilenetv3_small"))
    alpha = float(meta.get("alpha", meta.get("mu_weight", config.get("knn", {}).get("alpha", config.get("knn", {}).get("mu_weight", 1.0)))))
    m = float(meta.get("m", config.get("fcm", {}).get("m", 2.0)))
    error = float(meta.get("error", config.get("fcm", {}).get("error", 1e-5)))
    maxiter = int(meta.get("maxiter", config.get("fcm", {}).get("maxiter", 200)))
    seed = int(meta.get("random_state", config.get("dataset", {}).get("random_state", 42)))

    # Auto-deteksi mode ekstraktor dari dimensi input scaler jika berbeda
    if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
        dim = len(scaler.mean_)
        if dim <= 27 and feature_extractor != "handcrafted":
            feature_extractor = "handcrafted"
        elif dim == 576 and feature_extractor != "mobilenetv3_small":
            feature_extractor = "mobilenetv3_small"

    # 1. Ekstraksi Fitur
    raw_feature = extract_features(image_or_path, feature_extractor=feature_extractor, config=config)
    raw_feature_2d = raw_feature.reshape(1, -1)

    # 2. Skalasi dan Proyeksi PCA
    scaled_feature = scaler.transform(raw_feature_2d)
    pca_feature = pca.transform(scaled_feature)

    # 3. Hitung Derajat Keanggotaan FCM via fcm_predict_membership
    memberships = fcm_predict_membership(
        pca_feature, fcm_centers, m=m, error=error, maxiter=maxiter, seed=seed
    )[0]

    dominant_cluster = int(np.argmax(memberships))
    dominant_membership = float(memberships[dominant_cluster])

    # 4. Bentuk Input KNN: [PCA || alpha * mu]
    knn_input = np.hstack([pca_feature, (memberships * alpha).reshape(1, -1)])

    # 5. Prediksi Kelas via KNN
    predicted_label = knn.predict(knn_input)[0]

    class_probabilities = {}
    if hasattr(knn, "predict_proba"):
        try:
            probas = knn.predict_proba(knn_input)[0]
            if hasattr(knn, "classes_"):
                for cls, prob in zip(knn.classes_, probas):
                    class_probabilities[str(cls)] = float(prob)
        except Exception:
            pass

    # 6. Teks Interpretasi
    sorted_cluster_idx = np.argsort(memberships)[::-1]
    top_memberships_str = [
        f"Cluster {idx}: {memberships[idx]:.1%}"
        for idx in sorted_cluster_idx[:3]
    ]

    interp_text = (
        f"Citra diklasifikasikan sebagai kategori '{predicted_label}'. "
        f"Derajat keanggotaan fuzzy tertinggi berada pada Cluster {dominant_cluster} sebesar {dominant_membership:.2%}. "
        f"Distribusi keanggotaan fuzzy teratas: {', '.join(top_memberships_str)}."
    )

    return {
        "predicted_class": str(predicted_label),
        "dominant_cluster": dominant_cluster,
        "dominant_membership": dominant_membership,
        "fcm_memberships": memberships.tolist(),
        "class_probabilities": class_probabilities,
        "pca_features": pca_feature[0].tolist(),
        "pc1": float(pca_feature[0, 0]),
        "pc2": float(pca_feature[0, 1]) if pca_feature.shape[1] > 1 else 0.0,
        "raw_feature_dim": int(raw_feature.shape[0]),
        "alpha": alpha,
        "interpretation": interp_text,
    }


# ==============================================================================
# Pemuatan & Penyimpanan Model (save_models / load_models)
# ==============================================================================

def load_models(models_dir: Union[str, Path] = "models") -> Optional[Dict[str, Any]]:
    """
    Memuat 5 artefak model dari direktori models/:
    - scaler.pkl
    - pca.pkl
    - fcm_centers.npy
    - knn.pkl
    - meta.json

    Mengembalikan None secara ramah jika artefak belum lengkap.
    """
    dir_path = Path(models_dir)
    if not dir_path.is_dir():
        return None

    scaler_path = dir_path / "scaler.pkl"
    pca_path = dir_path / "pca.pkl"
    fcm_path = dir_path / "fcm_centers.npy"
    knn_path = dir_path / "knn.pkl"
    meta_path = dir_path / "meta.json"

    if not (scaler_path.is_file() and pca_path.is_file() and fcm_path.is_file() and knn_path.is_file()):
        return None

    try:
        if JOBLIB_AVAILABLE:
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_path)
            knn = joblib.load(knn_path)
        else:
            with open(scaler_path, "rb") as f:
                scaler = joblib.load(f)
            with open(pca_path, "rb") as f:
                pca = joblib.load(f)
            with open(knn_path, "rb") as f:
                knn = joblib.load(f)

        fcm_centers = np.load(fcm_path)

        meta = {}
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        class_names = meta.get("class_names", [])
        if not class_names and hasattr(knn, "classes_"):
            class_names = list(knn.classes_)

        m_val = float(meta.get("m", 2.0))
        fcm = FCM(n_clusters=len(fcm_centers), m=m_val)
        fcm.centers_ = fcm_centers

        return {
            "scaler": scaler,
            "pca": pca,
            "fcm_centers": fcm_centers,
            "fcm": fcm,
            "knn": knn,
            "meta": meta,
            "class_names": class_names,
            "config": meta.get("config", load_default_config()),
        }
    except Exception:
        return None


def save_models(
    models_dir: Union[str, Path] = "models",
    scaler: Any = None,
    pca: Any = None,
    fcm_centers: Any = None,
    knn: Any = None,
    meta: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    classes: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    feature_extractor: Optional[str] = None,
    alpha: Optional[float] = None,
    mu_weight: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """
    Menyimpan 5 artefak model ke direktori models/:
    - scaler.pkl
    - pca.pkl
    - fcm_centers.npy
    - knn.pkl
    - meta.json
    """
    dir_path = Path(models_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    meta_dict = meta.copy() if meta else {}
    if class_names is not None:
        meta_dict["class_names"] = class_names
    elif classes is not None:
        meta_dict["class_names"] = classes
    if feature_extractor is not None:
        meta_dict["feature_extractor"] = feature_extractor
    if alpha is not None:
        meta_dict["alpha"] = alpha
        meta_dict["mu_weight"] = alpha
    elif mu_weight is not None:
        meta_dict["alpha"] = mu_weight
        meta_dict["mu_weight"] = mu_weight

    saved_paths: Dict[str, str] = {}

    if scaler is not None:
        p = dir_path / "scaler.pkl"
        if JOBLIB_AVAILABLE:
            joblib.dump(scaler, p)
        else:
            with open(p, "wb") as f:
                joblib.dump(scaler, f)
        saved_paths["scaler"] = str(p)

    if pca is not None:
        p = dir_path / "pca.pkl"
        if JOBLIB_AVAILABLE:
            joblib.dump(pca, p)
        else:
            with open(p, "wb") as f:
                joblib.dump(pca, f)
        saved_paths["pca"] = str(p)

    if fcm_centers is not None:
        p = dir_path / "fcm_centers.npy"
        np.save(p, np.asarray(fcm_centers, dtype=np.float64))
        saved_paths["fcm_centers"] = str(p)

    if knn is not None:
        p = dir_path / "knn.pkl"
        if JOBLIB_AVAILABLE:
            joblib.dump(knn, p)
        else:
            with open(p, "wb") as f:
                joblib.dump(knn, f)
        saved_paths["knn"] = str(p)

    meta_dict = meta.copy() if meta else {}
    if config:
        meta_dict.setdefault("config", config)
    meta_dict.setdefault("saved_at", datetime.now().isoformat())

    lib_versions = {
        "numpy": np.__version__,
    }
    if SKLEARN_AVAILABLE:
        import sklearn
        lib_versions["scikit-learn"] = sklearn.__version__
    try:
        import skfuzzy
        lib_versions["scikit-fuzzy"] = skfuzzy.__version__
    except ImportError:
        pass
    try:
        import tensorflow as tf
        lib_versions["tensorflow"] = tf.__version__
    except ImportError:
        pass

    meta_dict["library_versions"] = lib_versions

    clean_meta = _to_json_serializable(meta_dict)
    meta_file = dir_path / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(clean_meta, f, indent=2)
    saved_paths["meta"] = str(meta_file)

    return saved_paths


def evaluate_pipeline(
    models: Dict[str, Any],
    X: Union[List[str], np.ndarray],
    y_true: Union[List[str], np.ndarray],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi pipeline pada dataset uji.
    """
    scaler = models["scaler"]
    pca = models["pca"]
    knn = models["knn"]
    fcm_centers = models["fcm_centers"]
    meta = models.get("meta", {})

    if config is None:
        config = models.get("config", load_default_config())

    feature_extractor = meta.get("feature_extractor", config.get("features", {}).get("feature_extractor", "mobilenetv3_small"))
    alpha = float(meta.get("alpha", meta.get("mu_weight", config.get("knn", {}).get("alpha", 1.0))))
    m = float(meta.get("m", config.get("fcm", {}).get("m", 2.0)))
    error = float(meta.get("error", config.get("fcm", {}).get("error", 1e-5)))
    maxiter = int(meta.get("maxiter", config.get("fcm", {}).get("maxiter", 200)))
    seed = int(meta.get("random_state", config.get("dataset", {}).get("random_state", 42)))

    if isinstance(X, list) and len(X) > 0 and isinstance(X[0], (str, Path)):
        X_feat = np.array([
            extract_features(p, feature_extractor=feature_extractor, config=config)
            for p in X
        ], dtype=np.float64)
    else:
        X_feat = np.asarray(X, dtype=np.float64)

    X_scaled = scaler.transform(X_feat)
    X_pca = pca.transform(X_scaled)

    mu_test = fcm_predict_membership(X_pca, fcm_centers, m=m, error=error, maxiter=maxiter, seed=seed)
    fpc = compute_fpc(mu_test)
    pe = compute_partition_entropy(mu_test)

    X_knn = np.hstack([X_pca, mu_test * alpha])
    y_pred = knn.predict(X_knn)

    class_names = models.get("class_names", sorted(list(np.unique(y_true))))
    metrics = compute_classification_metrics(np.asarray(y_true), y_pred, labels=class_names)

    return {
        "fpc": fpc,
        "pe": pe,
        "classification_metrics": metrics,
        "n_samples": len(X_pca),
    }
