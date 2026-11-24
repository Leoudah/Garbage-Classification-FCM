"""
Modul Pipeline Pelatihan & Inferensi (pipeline.py)
=================================================
Pipeline Terintegrasi Final Project TLF:
1. Dataset Loader dinamis dari subfolder kelas (berdasarkan nama folder, tanpa hardcode).
2. Stratified Train/Test Split (test_size=0.2, random_state=42) SEBELUM tahap fitting.
3. Ekstraksi Fitur (MobileNetV2 1280-dimensi atau Handcrafted ~26-dimensi).
4. StandardScaler: HANYA di-fit pada data train (scaler.fit(X_train)), lalu transform X_train & X_test.
5. PCA: HANYA di-fit pada data train terskala dengan target cumulative variance threshold (default 90%)
   dan batas maksimum komponen (max_components=50). Menghasilkan koordinat PC1-PC2.
6. FCM Manual: HANYA di-fit pada X_train_pca. Menghasilkan pusat cluster V, keanggotaan mu_train, FPC, PE, Silhouette.
7. Prediksi Keanggotaan FCM Test: Dihitung menggunakan fcm_predict_membership(X_test_pca, V, m).
8. Klasifikasi KNN: Dilatih pada fitur gabungan [PCA, mu * mu_weight] dengan label kelas asli.
9. Evaluasi Komprehensif: FPC, PE, Silhouette untuk FCM; Accuracy, Macro Precision, Macro Recall,
   Macro F1, dan Confusion Matrix untuk KNN.
10. Fungsi Komparasi:
    - compare_knn_variants: Membandingkan KNN biasa (PCA saja) vs KNN + mu FCM (PCA + mu).
    - compare_feature_extractors: Membandingkan MobileNetV2 vs Handcrafted.
11. Penyimpanan & Pemuatan Model (save_models / load_models) termasuk meta.json.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

# Coba import joblib, jika belum ada fallback ke pickle
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle as joblib
    JOBLIB_AVAILABLE = False

# Import modul internal
from .fcm import FCM, fcm_predict_membership, compute_fpc, compute_partition_entropy
from .features import (
    extract_features,
    extract_mobilenet_features,
    extract_handcrafted_features,
    get_feature_names,
)

# Coba import scikit-learn jika terinstal, sediakan fallback mandiri
try:
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
    from sklearn.metrics import (
        accuracy_score as sklearn_accuracy_score,
        precision_score as sklearn_precision_score,
        recall_score as sklearn_recall_score,
        f1_score as sklearn_f1_score,
        confusion_matrix as sklearn_confusion_matrix,
        silhouette_score as sklearn_silhouette_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# Fallback Classes & Functions (NumPy Murni)
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
    def __init__(self, n_components: int = 4, random_state: Optional[int] = 42):
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

        k = max(1, min(self.n_components, n_samples, n_features))
        self.components_ = vt[:k, :]
        exp_var = (s[:k] ** 2) / (n_samples - 1 if n_samples > 1 else 1)
        self.explained_variance_ = exp_var
        self.explained_variance_ratio_ = exp_var / (total_var if total_var > 0 else 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X - self.mean_, self.components_.T)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class _NumpyKNN:
    """Implementasi fallback KNeighborsClassifier berbasis NumPy."""
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyKNN":
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("KNN belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        diff = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        k = max(1, min(self.n_neighbors, self.X_train_.shape[0]))
        nn_indices = np.argsort(dists, axis=1)[:, :k]

        preds = []
        for row in nn_indices:
            neighbor_labels = self.y_train_[row]
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None or self.classes_ is None:
            raise RuntimeError("KNN belum di-fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        diff = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        k = max(1, min(self.n_neighbors, self.X_train_.shape[0]))
        nn_indices = np.argsort(dists, axis=1)[:, :k]

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for i, row in enumerate(nn_indices):
            neighbor_labels = self.y_train_[row]
            for lbl in neighbor_labels:
                if lbl in class_to_idx:
                    probas[i, class_to_idx[lbl]] += 1.0 / k
        return probas


def _get_standard_scaler():
    if SKLEARN_AVAILABLE:
        return SklearnStandardScaler()
    return _NumpyStandardScaler()


def _get_pca(n_components: int = 4, random_state: Optional[int] = 42):
    if SKLEARN_AVAILABLE:
        return SklearnPCA(n_components=n_components, random_state=random_state)
    return _NumpyPCA(n_components=n_components, random_state=random_state)


def _get_knn(n_neighbors: int = 5, metric: str = "euclidean"):
    if SKLEARN_AVAILABLE:
        return SklearnKNN(n_neighbors=n_neighbors, metric=metric)
    return _NumpyKNN(n_neighbors=n_neighbors, metric=metric)


def _stratified_split(
    X_items: List[Any],
    y_items: Union[np.ndarray, List[Any]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Any], List[Any], np.ndarray, np.ndarray]:
    """Melakukan stratified split train/test secara deterministik."""
    if SKLEARN_AVAILABLE:
        try:
            return sklearn_train_test_split(
                X_items, y_items, test_size=test_size, stratify=y_items, random_state=random_state
            )
        except Exception:
            pass

    # Fallback NumPy stratified split
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


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Menghitung Accuracy, Macro Precision, Macro Recall, Macro F1, dan Confusion Matrix.
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
            prec = float(sklearn_precision_score(y_t, y_p, average="macro", zero_division=0))
            rec = float(sklearn_recall_score(y_t, y_p, average="macro", zero_division=0))
            f1 = float(sklearn_f1_score(y_t, y_p, average="macro", zero_division=0))
            cm = sklearn_confusion_matrix(y_t, y_p, labels=all_labels).tolist()
            return {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "confusion_matrix": cm,
                "labels": all_labels.tolist(),
            }
        except Exception:
            pass

    # Fallback NumPy murni
    acc = float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []

    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)

    for t, p in zip(y_t, y_p):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    for lbl in all_labels:
        tp = np.sum((y_t == lbl) & (y_p == lbl))
        fp = np.sum((y_t != lbl) & (y_p == lbl))
        fn = np.sum((y_t == lbl) & (y_p != lbl))

        p_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0

        precisions.append(p_c)
        recalls.append(r_c)
        f1s.append(f_c)

    return {
        "accuracy": acc,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels.tolist(),
    }


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


def _compute_silhouette(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Menghitung Silhouette Score pada ruang data/PCA."""
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    if len(unique_labels) <= 1 or len(unique_labels) >= n_samples:
        return None

    if SKLEARN_AVAILABLE:
        try:
            return float(sklearn_silhouette_score(X, labels))
        except Exception:
            pass

    # Fallback NumPy aproksimasi
    try:
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        a_vals = np.zeros(n_samples)
        b_vals = np.full(n_samples, np.inf)

        for i in range(n_samples):
            same_cluster = labels == labels[i]
            same_count = np.sum(same_cluster)
            a_vals[i] = np.sum(dists[i, same_cluster]) / (same_count - 1) if same_count > 1 else 0.0

            for lbl in unique_labels:
                if lbl != labels[i]:
                    other_cluster = labels == lbl
                    mean_dist_other = np.mean(dists[i, other_cluster])
                    if mean_dist_other < b_vals[i]:
                        b_vals[i] = mean_dist_other

        max_ab = np.maximum(a_vals, b_vals)
        s_i = np.where(max_ab > 0, (b_vals - a_vals) / max_ab, 0.0)
        return float(np.mean(s_i))
    except Exception:
        return None


# ==============================================================================
# Pemuatan Dataset Dinamis dari Direktori
# ==============================================================================

def load_dataset_from_directory(
    data_dir: Union[str, Path],
    samples_per_class: int = 100,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Memuat dataset citra dari subfolder kelas secara dinamis (tanpa hardcode nama kelas).

    Parameters
    ----------
    data_dir : str atau Path
        Direktori induk yang berisi subfolder kelas (misal: data/Garbage_Dataset_Classification/images).
    samples_per_class : int, default=100
        Jumlah sampel maksimum per kelas yang diambil untuk efisiensi komputasi.
    random_state : int, default=42
        Seed untuk pengacakan pemilihan sampel gambar.

    Returns
    -------
    tuple of (filepaths, labels, class_names)
        filepaths : list of str (jalur file gambar yang terpilih)
        labels : np.ndarray of shape (n_samples,) berisi label string nama kelas
        class_names : list of str (daftar nama kelas unik yang ditemukan)
    """
    base_path = Path(data_dir)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {data_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rng = np.random.RandomState(random_state)

    filepaths: List[str] = []
    labels: List[str] = []
    class_names: List[str] = []

    # Ambil seluruh subdirektori sebagai kelas secara dinamis
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"Tidak ada subfolder kelas ditemukan di {data_dir}")

    for sub in subdirs:
        c_name = sub.name
        class_names.append(c_name)

        # Temukan file citra dalam subfolder
        img_files = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        img_files = sorted(img_files, key=lambda x: x.name)

        if len(img_files) == 0:
            continue

        # Sampling subset sampel per kelas
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
            "samples_per_class": 100,
            "test_size": 0.2,
            "random_state": 42,
        },
        "features": {
            "feature_extractor": "mobilenetv2",
            "image_size": [224, 224],
            "use_edge_density": False,
        },
        "pca": {
            "variance_threshold": 0.90,
            "max_components": 50,
        },
        "fcm": {
            "c": 6,
            "m": 2.0,
            "max_iter": 150,
            "epsilon": 0.00001,
            "random_state": 42,
        },
        "knn": {
            "n_neighbors": 5,
            "metric": "euclidean",
            "mu_weight": 1.0,
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
        },
    }


# ==============================================================================
# Pipeline Training Lengkap
# ==============================================================================

def fit_pipeline(
    X: Union[List[str], np.ndarray],
    y: Optional[Union[List[str], np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    feature_extractor: Optional[str] = None,
    y_true: Optional[Union[List[str], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Melatih seluruh pipeline PCA + FCM + KNN:
    a. Stratified train/test split (test_size=0.2, random_state=42) SEBELUM fitting.
    b. Ekstraksi fitur (MobileNetV2 atau Handcrafted).
    c. StandardScaler: HANYA di-fit pada X_train_feat.
    d. PCA: HANYA di-fit pada X_train_scaled (variance threshold 90%, max 50 komponen).
    e. FCM: HANYA di-fit pada X_train_pca -> pusat V, mu_train, FPC, PE, Silhouette.
    f. Prediksi mu_test melalui fcm_predict_membership(X_test_pca, V, m).
    g. Latih KNN pada gabungan [X_train_pca, mu_train * mu_weight] dengan label asli.
    h. Evaluasi KNN pada data test dan komparasi varian (PCA saja vs PCA + mu FCM).

    Parameters
    ----------
    X : list of str (image paths) atau np.ndarray 2D fitur
    y : array-like label kelas asli (wajib untuk stratified split & evaluasi)
    config : dict, optional
    feature_extractor : str, optional ("mobilenetv2" | "handcrafted")

    Returns
    -------
    dict
        Objek model terlatih, data proyeksi PCA, metrik evaluasi train/test, dan komparasi varian.
    """
    if y is None and y_true is not None:
        y = y_true

    if config is None:
        config = load_default_config()

    if feature_extractor is None:
        feature_extractor = config.get("features", {}).get("feature_extractor", "mobilenetv2")

    # Konfigurasi parameter
    dataset_cfg = config.get("dataset", {})
    test_size = float(dataset_cfg.get("test_size", 0.2))
    random_state = int(dataset_cfg.get("random_state", 42))

    pca_cfg = config.get("pca", {})
    var_thresh = float(pca_cfg.get("variance_threshold", 0.90))
    max_components = int(pca_cfg.get("max_components", 50))

    fcm_cfg = config.get("fcm", {})
    c = int(fcm_cfg.get("c", 6))
    m = float(fcm_cfg.get("m", 2.0))
    max_iter = int(fcm_cfg.get("max_iter", 150))
    epsilon = float(fcm_cfg.get("epsilon", 1e-5))

    knn_cfg = config.get("knn", {})
    k_neighbors = int(knn_cfg.get("n_neighbors", 5))
    metric = str(knn_cfg.get("metric", "euclidean"))
    mu_weight = float(knn_cfg.get("mu_weight", 1.0))

    # Pastikan label y tersedia
    if y is None:
        if isinstance(X, np.ndarray):
            y_arr = np.zeros(len(X), dtype=int)
        else:
            raise ValueError("Label y wajib disediakan untuk alur supervised KNN dan stratified split.")
    else:
        y_arr = np.asarray(y)

    # 1. Stratified Train/Test Split SEBELUM fitting model apapun
    X_train_raw, X_test_raw, y_train, y_test = _stratified_split(
        X, y_arr, test_size=test_size, random_state=random_state
    )

    class_names = sorted(list(np.unique(y_arr)))

    # 2. Ekstraksi Fitur
    def _needs_feature_extraction(sample: Any) -> bool:
        from PIL import Image
        if isinstance(sample, (str, Path, Image.Image)):
            return True
        if isinstance(sample, np.ndarray) and sample.ndim >= 2:
            return True
        return False

    if (isinstance(X_train_raw, list) and len(X_train_raw) > 0 and _needs_feature_extraction(X_train_raw[0])) or \
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

    # 3. StandardScaler HANYA di-fit pada data train
    scaler = _get_standard_scaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # 4. PCA HANYA di-fit pada data train yang telah diskalakan
    n_samples_train, n_features_raw = X_train_scaled.shape
    max_possible_comp = min(max_components, n_samples_train - 1 if n_samples_train > 1 else 1, n_features_raw)

    # Fit PCA awal untuk menghitung varians kumulatif
    pca_initial = _get_pca(n_components=max_possible_comp, random_state=random_state)
    pca_initial.fit(X_train_scaled)

    cum_var = np.cumsum(pca_initial.explained_variance_ratio_)
    idx_reached = np.where(cum_var >= var_thresh)[0]
    if len(idx_reached) > 0:
        opt_k = idx_reached[0] + 1
    else:
        opt_k = max_possible_comp

    # Batasi minimal 2 komponen agar PC1 dan PC2 selalu ada untuk visualisasi scatter
    opt_k = max(2, min(opt_k, max_possible_comp))

    pca = _get_pca(n_components=opt_k, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 5. FCM Manual HANYA di-fit pada X_train_pca
    fcm_c = min(c, len(X_train_pca))
    fcm = FCM(
        n_clusters=fcm_c,
        m=m,
        max_iter=max_iter,
        error=epsilon,
        random_state=random_state,
    )
    fcm.fit(X_train_pca)

    fcm_centers = fcm.centers_
    mu_train = fcm.u_
    fpc_train = fcm.fpc_
    pe_train = fcm.pe_
    fcm_labels_train = fcm.labels_
    silhouette_train = _compute_silhouette(X_train_pca, fcm_labels_train)

    # 6. Hitung derajat keanggotaan mu_test pada data uji via fcm_predict_membership
    mu_test = fcm_predict_membership(X_test_pca, fcm_centers, m=m)
    fpc_test = compute_fpc(mu_test)
    pe_test = compute_partition_entropy(mu_test)

    # 7. Pembentukan Fitur untuk KNN: Gabungan [fitur_pca, mu * mu_weight]
    X_train_knn = np.hstack([X_train_pca, mu_train * mu_weight])
    X_test_knn = np.hstack([X_test_pca, mu_test * mu_weight])

    # Latih KNN dengan label kelas asli
    k_actual = max(1, min(k_neighbors, len(X_train_knn) - 1 if len(X_train_knn) > 1 else 1))
    knn = _get_knn(n_neighbors=k_actual, metric=metric)
    knn.fit(X_train_knn, y_train)

    # Prediksi KNN pada data uji
    y_pred_test = knn.predict(X_test_knn)
    knn_test_metrics = compute_classification_metrics(y_test, y_pred_test, labels=class_names)

    # 8. Komparasi Varian: KNN Biasa (PCA Saja) vs KNN + mu FCM (PCA + mu)
    variant_comparison = compare_knn_variants(
        X_train_pca=X_train_pca,
        mu_train=mu_train,
        y_train=y_train,
        X_test_pca=X_test_pca,
        mu_test=mu_test,
        y_test=y_test,
        k=k_actual,
        metric=metric,
        mu_weight=mu_weight,
        class_names=class_names,
    )

    result = {
        "scaler": scaler,
        "pca": pca,
        "fcm": fcm,
        "knn": knn,
        "fcm_centers": fcm_centers,
        "class_names": class_names,
        "config": config,
        "classes": class_names,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "mu_train": mu_train,
        "mu_test": mu_test,
        "y_train": y_train,
        "y_test": y_test,
        "meta": {
            "class_names": class_names,
            "feature_extractor": feature_extractor,
            "mu_weight": mu_weight,
            "n_components": opt_k,
            "variance_threshold": var_thresh,
            "c": fcm_c,
            "m": m,
            "k": k_actual,
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
                "objective_history": fcm.objective_history_,
            },
            "fcm_test": {
                "fpc": fpc_test,
                "pe": pe_test,
            },
            "pca": {
                "n_components": opt_k,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
            },
            "knn_test": knn_test_metrics,
            "variant_comparison": variant_comparison,
        },
    }

    return result


# ==============================================================================
# Fungsi Komparasi Varian & Fitur
# ==============================================================================

def compare_knn_variants(
    X_train_pca: np.ndarray,
    mu_train: np.ndarray,
    y_train: np.ndarray,
    X_test_pca: np.ndarray,
    mu_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
    mu_weight: float = 1.0,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Membandingkan performa dua varian klasifikasi KNN:
    - Varian 1 (Baseline): KNN Biasa (hanya fitur PCA).
    - Varian 2 (Proposed): KNN + Derajat Keanggotaan FCM (fitur PCA + mu * mu_weight).
    """
    k_actual = max(1, min(k, len(X_train_pca) - 1 if len(X_train_pca) > 1 else 1))

    # Varian 1: PCA saja
    knn_pca = _get_knn(n_neighbors=k_actual, metric=metric)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_test_pca)
    metrics_pca = compute_classification_metrics(y_test, y_pred_pca, labels=class_names)

    # Varian 2: PCA + FCM mu
    X_tr_hybrid = np.hstack([X_train_pca, mu_train * mu_weight])
    X_te_hybrid = np.hstack([X_test_pca, mu_test * mu_weight])
    knn_hybrid = _get_knn(n_neighbors=k_actual, metric=metric)
    knn_hybrid.fit(X_tr_hybrid, y_train)
    y_pred_hybrid = knn_hybrid.predict(X_te_hybrid)
    metrics_hybrid = compute_classification_metrics(y_test, y_pred_hybrid, labels=class_names)

    acc_diff = metrics_hybrid["accuracy"] - metrics_pca["accuracy"]
    f1_diff = metrics_hybrid["f1_macro"] - metrics_pca["f1_macro"]

    return {
        "knn_pca_only": metrics_pca,
        "knn_pca_plus_fcm": metrics_hybrid,
        "delta": {
            "accuracy": float(acc_diff),
            "f1_macro": float(f1_diff),
            "improved": bool(acc_diff >= 0),
        },
    }


def compare_feature_extractors(
    images_train: List[Any],
    y_train: np.ndarray,
    images_test: List[Any],
    y_test: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Membandingkan dua metode ekstraksi fitur: MobileNetV2 (1280 dimensi) vs Handcrafted HSV+GLCM (~26 dimensi).
    """
    if config is None:
        config = load_default_config()

    results: Dict[str, Any] = {}

    # 1. Ekstraktor Handcrafted
    try:
        cfg_hand = dict(config)
        cfg_hand["features"] = dict(config.get("features", {}))
        cfg_hand["features"]["feature_extractor"] = "handcrafted"

        res_hand = fit_pipeline(
            X=images_train + images_test,
            y=np.concatenate([y_train, y_test]),
            config=cfg_hand,
            feature_extractor="handcrafted",
        )
        results["handcrafted"] = {
            "accuracy": res_hand["metrics"]["knn_test"]["accuracy"],
            "f1_macro": res_hand["metrics"]["knn_test"]["f1_macro"],
            "fpc": res_hand["metrics"]["fcm_train"]["fpc"],
            "pe": res_hand["metrics"]["fcm_train"]["pe"],
            "n_components": res_hand["metrics"]["pca"]["n_components"],
        }
    except Exception as e:
        results["handcrafted"] = {"error": str(e)}

    # 2. Ekstraktor MobileNetV2
    try:
        cfg_mob = dict(config)
        cfg_mob["features"] = dict(config.get("features", {}))
        cfg_mob["features"]["feature_extractor"] = "mobilenetv2"

        res_mob = fit_pipeline(
            X=images_train + images_test,
            y=np.concatenate([y_train, y_test]),
            config=cfg_mob,
            feature_extractor="mobilenetv2",
        )
        results["mobilenetv2"] = {
            "accuracy": res_mob["metrics"]["knn_test"]["accuracy"],
            "f1_macro": res_mob["metrics"]["knn_test"]["f1_macro"],
            "fpc": res_mob["metrics"]["fcm_train"]["fpc"],
            "pe": res_mob["metrics"]["fcm_train"]["pe"],
            "n_components": res_mob["metrics"]["pca"]["n_components"],
        }
    except Exception as e:
        results["mobilenetv2"] = {"error": str(e)}

    return results


# ==============================================================================
# Inferensi Citra Tunggal (Predict)
# ==============================================================================

def transform(
    images_or_features: Union[np.ndarray, List[Any], str, Path],
    scaler: Any = None,
    pca: Any = None,
    config: Optional[Dict[str, Any]] = None,
    feature_extractor: str = "mobilenetv2",
) -> np.ndarray:
    """
    Mentransformasikan citra atau fitur ke dalam ruang PCA.
    """
    if scaler is None or pca is None:
        models = load_models()
        if models is None:
            raise RuntimeError("Model scaler dan pca belum ditemukan di models/.")
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
    1. Ekstraksi fitur (MobileNetV2 / Handcrafted).
    2. Transformasi StandardScaler.
    3. Proyeksi ke ruang PCA.
    4. Hitung derajat keanggotaan mu_ij via fcm_predict_membership terhadap fcm_centers.
    5. Bentuk fitur input KNN: [PCA, mu * mu_weight].
    6. Prediksi kelas dan probabilitas dengan KNN.
    7. Susun interpretasi komprehensif dalam Bahasa Indonesia.

    Parameters
    ----------
    image_or_path : str, Path, atau citra
    models : dict, optional (dari fit_pipeline atau load_models).
    config : dict, optional

    Returns
    -------
    dict
        Hasil prediksi kelas, probabilitas, derajat keanggotaan fuzzy, koordinat PCA, dan teks interpretasi.
    """
    if models is None:
        models = load_models()
        if models is None:
            raise RuntimeError(
                "Model belum dilatih atau file di folder models/ belum lengkap. "
                "Silakan jalankan notebook di Google Colab atau lakukan latih ulang via menu 'Proses'."
            )

    scaler = models["scaler"]
    pca = models["pca"]
    fcm_centers = models["fcm_centers"]
    knn = models["knn"]
    meta = models.get("meta", {})
    class_names = models.get("class_names", meta.get("class_names", []))

    if config is None:
        config = models.get("config", load_default_config())

    feature_extractor = meta.get("feature_extractor", config.get("features", {}).get("feature_extractor", "mobilenetv2"))
    mu_weight = float(meta.get("mu_weight", config.get("knn", {}).get("mu_weight", 1.0)))
    m = float(meta.get("m", config.get("fcm", {}).get("m", 2.0)))

    # Auto-deteksi mode ekstraktor dari dimensi scaler jika tersedia
    if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
        dim = len(scaler.mean_)
        if dim <= 27 and feature_extractor != "handcrafted":
            feature_extractor = "handcrafted"
        elif dim == 1280 and feature_extractor != "mobilenetv2":
            feature_extractor = "mobilenetv2"

    # 1. Ekstraksi Fitur
    raw_feature = extract_features(image_or_path, feature_extractor=feature_extractor, config=config)
    raw_feature_2d = raw_feature.reshape(1, -1)

    # 2. Skalasi dan Proyeksi PCA
    scaled_feature = scaler.transform(raw_feature_2d)
    pca_feature = pca.transform(scaled_feature)

    # 3. Hitung Derajat Keanggotaan FCM menggunakan fcm_predict_membership
    memberships = fcm_predict_membership(pca_feature, fcm_centers, m=m)[0]

    dominant_cluster = int(np.argmax(memberships))
    dominant_membership = float(memberships[dominant_cluster])

    # 4. Bentuk Input KNN: [PCA, mu * mu_weight]
    knn_input = np.hstack([pca_feature, (memberships * mu_weight).reshape(1, -1)])

    # 5. Prediksi Kelas via KNN
    predicted_label = knn.predict(knn_input)[0]

    # Ambil probabilitas jika tersedia
    class_probabilities = {}
    if hasattr(knn, "predict_proba"):
        try:
            probas = knn.predict_proba(knn_input)[0]
            if hasattr(knn, "classes_"):
                for cls, prob in zip(knn.classes_, probas):
                    class_probabilities[str(cls)] = float(prob)
        except Exception:
            pass

    # 6. Susun Teks Interpretasi Ramah
    sorted_cluster_idx = np.argsort(memberships)[::-1]
    top_memberships_str = [
        f"Cluster {idx}: {memberships[idx]:.1%}"
        for idx in sorted_cluster_idx[:3]
    ]

    interp_text = (
        f"Citra berhasil diidentifikasi sebagai kategori '{predicted_label}'. "
        f"Pada ruang proyeksi fuzzy, sampel memiliki derajat keanggotaan tertinggi pada Cluster {dominant_cluster} "
        f"sebesar {dominant_membership:.2%}. Komposisi keanggotaan fuzzy teratas: {', '.join(top_memberships_str)}."
    )

    membership_dict = {
        f"Cluster_{idx}": float(memberships[idx])
        for idx in range(len(memberships))
    }

    return {
        "predicted_class": str(predicted_label),
        "dominant_cluster": dominant_cluster,
        "dominant_membership": dominant_membership,
        "fcm_memberships": memberships.tolist(),
        "membership_dict": membership_dict,
        "class_probabilities": class_probabilities,
        "pca_features": pca_feature[0].tolist(),
        "pc1": float(pca_feature[0, 0]),
        "pc2": float(pca_feature[0, 1]) if pca_feature.shape[1] > 1 else 0.0,
        "raw_feature_dim": int(raw_feature.shape[0]),
        "interpretation": interp_text,
    }


# ==============================================================================
# Pemuatan & Penyimpanan Model (save_models / load_models)
# ==============================================================================

def load_models(models_dir: Union[str, Path] = "models") -> Optional[Dict[str, Any]]:
    """
    Memuat artefak model lengkap dari direktori models/:
    - scaler.pkl
    - pca.pkl
    - fcm_centers.npy
    - knn.pkl
    - meta.json (opsional jika ada)

    Jika salah satu file esensial belum tersedia, fungsi mengembalikan None secara ramah
    tanpa menimbulkan error/crash pada aplikasi Streamlit.
    """
    dir_path = Path(models_dir)
    if not dir_path.is_dir():
        return None

    scaler_path = dir_path / "scaler.pkl"
    pca_path = dir_path / "pca.pkl"
    fcm_path = dir_path / "fcm_centers.npy"
    knn_path = dir_path / "knn.pkl"
    meta_path = dir_path / "meta.json"

    # Periksa kelengkapan file biner model utama
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

        # Wrapper FCM untuk kemudahan akses
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
    mu_weight: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """
    Menyimpan seluruh artefak model dan metadata ke direktori models/:
    - scaler.pkl
    - pca.pkl
    - fcm_centers.npy
    - knn.pkl
    - meta.json
    """
    dir_path = Path(models_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    if meta is None:
        meta = {}
    if class_names is not None:
        meta["class_names"] = class_names
    elif classes is not None:
        meta["class_names"] = classes
    if feature_extractor is not None:
        meta["feature_extractor"] = feature_extractor
    if mu_weight is not None:
        meta["mu_weight"] = mu_weight

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

    # Simpan metadata JSON
    meta_dict = meta.copy() if meta else {}
    if config:
        meta_dict.setdefault("config", config)
    meta_dict.setdefault("saved_at", datetime.now().isoformat())

    # Informasi versi library untuk reproduktibilitas
    lib_versions = {
        "numpy": np.__version__,
    }
    if SKLEARN_AVAILABLE:
        import sklearn
        lib_versions["scikit-learn"] = sklearn.__version__
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
    Mengevaluasi pipeline pada dataset uji yang diberikan.
    """
    scaler = models["scaler"]
    pca = models["pca"]
    knn = models["knn"]
    fcm_centers = models["fcm_centers"]
    meta = models.get("meta", {})

    if config is None:
        config = models.get("config", load_default_config())

    feature_extractor = meta.get("feature_extractor", config.get("features", {}).get("feature_extractor", "mobilenetv2"))
    mu_weight = float(meta.get("mu_weight", config.get("knn", {}).get("mu_weight", 1.0)))
    m = float(meta.get("m", config.get("fcm", {}).get("m", 2.0)))

    # Auto-deteksi mode ekstraktor dari dimensi scaler jika tersedia
    if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
        dim = len(scaler.mean_)
        if dim <= 27 and feature_extractor != "handcrafted":
            feature_extractor = "handcrafted"
        elif dim == 1280 and feature_extractor != "mobilenetv2":
            feature_extractor = "mobilenetv2"

    if isinstance(X, list) and len(X) > 0 and isinstance(X[0], (str, Path)):
        X_feat = np.array([
            extract_features(p, feature_extractor=feature_extractor, config=config)
            for p in X
        ], dtype=np.float64)
    else:
        X_feat = np.asarray(X, dtype=np.float64)

    X_scaled = scaler.transform(X_feat)
    X_pca = pca.transform(X_scaled)

    mu_test = fcm_predict_membership(X_pca, fcm_centers, m=m)
    fpc = compute_fpc(mu_test)
    pe = compute_partition_entropy(mu_test)

    X_knn = np.hstack([X_pca, mu_test * mu_weight])
    y_pred = knn.predict(X_knn)

    class_names = models.get("class_names", sorted(list(np.unique(y_true))))
    metrics = compute_classification_metrics(np.asarray(y_true), y_pred, labels=class_names)

    return {
        "fpc": fpc,
        "pe": pe,
        "classification_metrics": metrics,
        "n_samples": len(X_pca),
    }
