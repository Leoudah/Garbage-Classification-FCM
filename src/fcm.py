"""
Modul Fuzzy C-Means (FCM) Berbasis scikit-fuzzy (fcm.py)
========================================================
Implementasi wrapper algoritma clustering Fuzzy C-Means menggunakan library scikit-fuzzy
(skfuzzy.cluster.cmeans dan skfuzzy.cluster.cmeans_predict), dilengkapi fallback NumPy murni
jika scikit-fuzzy belum terpasang di environment lokal.

PENTING - Konvensi Bentuk Matriks scikit-fuzzy:
- skfuzzy.cluster.cmeans menerima data dengan bentuk (n_features, n_samples) -> X_pca.T
- skfuzzy.cluster.cmeans mengembalikan matriks keanggotaan u berukuran (c, n_samples)
- Matriks u ditransposisikan kembali (u.T) menjadi (n_samples, c) agar sesuai dengan standar
  machine learning dan format fitur gabungan [PCA || alpha * mu].

Metrik Kualitas:
1. P (Nilai Fungsi Objektif Akhir): jm[-1]
2. n_iter (Iterasi Hingga Konvergen): p
3. FPC (Fuzzy Partition Coefficient): rentang [1/c, 1.0], semakin tinggi semakin tegas
4. PE (Partition Entropy): -mean(sum(u * ln(u))), semakin kecil semakin terpisah dengan baik
5. Silhouette Score: evaluasi pemisahan cluster berbasis jarak Euclidean pada ruang PCA
"""

from typing import List, Optional, Tuple, Union
import numpy as np

# Cek ketersediaan scikit-fuzzy
try:
    import skfuzzy.cluster as skf_cluster
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False

# Cek ketersediaan scikit-learn untuk silhouette score
try:
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# Fallback scikit-fuzzy murni NumPy (Jika skfuzzy belum terpasang)
# ==============================================================================

def _cmeans_numpy(
    data: np.ndarray,
    c: int,
    m: float = 2.0,
    error: float = 1e-5,
    maxiter: int = 200,
    metric: str = "euclidean",
    init: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], int, float]:
    """
    Implementasi fallback cmeans dengan antarmuka dan perilaku identik terhadap skfuzzy.cluster.cmeans.
    
    Parameters
    ----------
    data : np.ndarray of shape (n_features, n_samples)
    c : int (jumlah cluster)
    m : float (fuzziness exponent, default 2.0)
    error : float (ambang henti / epsilon, default 1e-5)
    maxiter : int (batas maksimum iterasi, default 200)
    
    Returns
    -------
    cntr : (c, n_features)
    u : (c, n_samples)
    u0 : (c, n_samples)
    d : (c, n_samples)
    jm : list of float
    p : int
    fpc : float
    """
    n_features, n_samples = data.shape
    X = data.T  # (n_samples, n_features)

    rng = np.random.RandomState(seed)
    if init is None:
        u0 = rng.uniform(0.01, 1.0, size=(c, n_samples))
        u0 = u0 / np.sum(u0, axis=0, keepdims=True)
    else:
        u0 = np.asarray(init, dtype=np.float64)

    u = u0.copy()
    jm: List[float] = []
    p = 0
    cntr = np.zeros((c, n_features), dtype=np.float64)
    d = np.zeros((c, n_samples), dtype=np.float64)

    power = 2.0 / (m - 1.0)

    for it in range(maxiter):
        p = it + 1
        # 1. Hitung pusat cluster cntr: (c, n_features)
        u_m = u ** m  # (c, n_samples)
        numerator = np.dot(u_m, X)  # (c, n_features)
        denominator = np.sum(u_m, axis=1, keepdims=True)  # (c, 1)
        denominator = np.where(denominator == 0, 1e-12, denominator)
        cntr = numerator / denominator

        # 2. Hitung jarak Euclidean d: (c, n_samples)
        diff = X[:, np.newaxis, :] - cntr[np.newaxis, :, :]  # (n_samples, c, n_features)
        dist_matrix = np.linalg.norm(diff, axis=2)  # (n_samples, c)
        d = dist_matrix.T  # (c, n_samples)

        # 3. Hitung fungsi objektif J
        j_val = float(np.sum((u ** m) * (d ** 2)))
        jm.append(j_val)

        # 4. Pembaruan u dengan penanganan divide-by-zero
        u_new = np.zeros((c, n_samples), dtype=np.float64)
        zero_mask = d < 1e-12
        cols_with_zeros = zero_mask.any(axis=0)

        normal_cols = ~cols_with_zeros
        if np.any(normal_cols):
            d_norm = d[:, normal_cols]
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_d = d_norm ** (-power)
            sum_inv = np.sum(inv_d, axis=0, keepdims=True)
            sum_inv = np.where(sum_inv == 0, 1e-12, sum_inv)
            u_new[:, normal_cols] = inv_d / sum_inv

        if np.any(cols_with_zeros):
            for col_idx in np.where(cols_with_zeros)[0]:
                zero_rows = np.where(zero_mask[:, col_idx])[0]
                u_new[:, col_idx] = 0.0
                u_new[zero_rows, col_idx] = 1.0 / len(zero_rows)

        # Cek konvergensi
        if np.max(np.abs(u_new - u)) < error:
            u = u_new
            break
        u = u_new

    fpc = float(np.sum(u ** 2) / n_samples)
    return cntr, u, u0, d, jm, p, fpc


def _cmeans_predict_numpy(
    test_data: np.ndarray,
    cntr_trained: np.ndarray,
    m: float = 2.0,
    error: float = 1e-5,
    maxiter: int = 200,
    metric: str = "euclidean",
    init: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float], int, float]:
    """
    Implementasi fallback cmeans_predict identik terhadap skfuzzy.cluster.cmeans_predict.
    """
    n_features, n_samples = test_data.shape
    c = cntr_trained.shape[0]
    X_test = test_data.T  # (n_samples, n_features)

    # Hitung jarak langsung ke pusat cluster yang sudah dilatih
    diff = X_test[:, np.newaxis, :] - cntr_trained[np.newaxis, :, :]  # (n_samples, c, n_features)
    dist_matrix = np.linalg.norm(diff, axis=2)  # (n_samples, c)
    d = dist_matrix.T  # (c, n_samples)

    power = 2.0 / (m - 1.0)
    u = np.zeros((c, n_samples), dtype=np.float64)

    zero_mask = d < 1e-12
    cols_with_zeros = zero_mask.any(axis=0)

    normal_cols = ~cols_with_zeros
    if np.any(normal_cols):
        d_norm = d[:, normal_cols]
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_d = d_norm ** (-power)
        sum_inv = np.sum(inv_d, axis=0, keepdims=True)
        sum_inv = np.where(sum_inv == 0, 1e-12, sum_inv)
        u[:, normal_cols] = inv_d / sum_inv

    if np.any(cols_with_zeros):
        for col_idx in np.where(cols_with_zeros)[0]:
            zero_rows = np.where(zero_mask[:, col_idx])[0]
            u[:, col_idx] = 0.0
            u[zero_rows, col_idx] = 1.0 / len(zero_rows)

    jm_val = float(np.sum((u ** m) * (d ** 2)))
    fpc = float(np.sum(u ** 2) / n_samples)
    return u, u.copy(), d, [jm_val], 1, fpc


# ==============================================================================
# Fungsi Metrik Kualitas FCM
# ==============================================================================

def compute_fpc(u: np.ndarray) -> float:
    """
    Menghitung Fuzzy Partition Coefficient (FPC).
    u berukuran (n_samples, c).
    """
    n_samples = u.shape[0]
    if n_samples == 0:
        return 0.0
    return float(np.sum(u ** 2) / n_samples)


def compute_partition_entropy(u: np.ndarray) -> float:
    """
    Menghitung Partition Entropy (PE).
    PE = - (1 / n) * sum_i sum_j (u_ij * ln(u_ij)) untuk u_ij > 0.
    """
    n_samples = u.shape[0]
    if n_samples == 0:
        return 0.0
    valid_mask = u > 1e-12
    val = np.zeros_like(u)
    val[valid_mask] = u[valid_mask] * np.log(u[valid_mask])
    pe = - float(np.sum(val) / n_samples)
    return max(0.0, pe)


# Alias
compute_pe = compute_partition_entropy


def compute_fcm_silhouette(X: np.ndarray, u: np.ndarray) -> Optional[float]:
    """
    Menghitung Silhouette Score pada ruang data/PCA dengan label cluster crisp = argmax(u).
    """
    if X.shape[0] < 2:
        return None
    labels = np.argmax(u, axis=1)
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= X.shape[0]:
        return None

    if SKLEARN_AVAILABLE:
        try:
            return float(sklearn_silhouette_score(X, labels))
        except Exception:
            pass

    # Fallback NumPy murni
    try:
        n_samples = X.shape[0]
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        a_vals = np.zeros(n_samples)
        b_vals = np.full(n_samples, np.inf)

        for i in range(n_samples):
            same_mask = labels == labels[i]
            same_cnt = np.sum(same_mask)
            a_vals[i] = np.sum(dists[i, same_mask]) / (same_cnt - 1) if same_cnt > 1 else 0.0

            for lbl in unique_labels:
                if lbl != labels[i]:
                    other_mask = labels == lbl
                    mean_other = np.mean(dists[i, other_mask])
                    if mean_other < b_vals[i]:
                        b_vals[i] = mean_other

        max_ab = np.maximum(a_vals, b_vals)
        s_i = np.where(max_ab > 0, (b_vals - a_vals) / max_ab, 0.0)
        return float(np.mean(s_i))
    except Exception:
        return None


# ==============================================================================
# Fungsi Standalone fcm_predict_membership
# ==============================================================================

def fcm_predict_membership(
    X: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
    error: float = 1e-5,
    maxiter: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Menghitung derajat keanggotaan mu_ij untuk data baru / data uji berdasarkan
    jarak ke pusat cluster V hasil training FCM (menggunakan skfuzzy.cluster.cmeans_predict).

    BENTUK MATRIKS:
    - Input X: (n_samples, n_features)
    - Ditransformasi ke skfuzzy: X.T -> (n_features, n_samples)
    - Output skfuzzy u: (c, n_samples)
    - Ditransposisikan kembali: u.T -> (n_samples, c)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features) atau 1D (n_features,)
    centers : np.ndarray of shape (n_clusters, n_features)
    m : float, default=2.0
    error : float, default=1e-5
    maxiter : int, default=200
    seed : int, default=42

    Returns
    -------
    np.ndarray of shape (n_samples, n_clusters)
        Matriks derajat keanggotaan ternormalisasi (sum_j mu_ij = 1.0).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if centers_arr.ndim != 2:
        raise ValueError(f"centers harus berupa matriks 2D (c, n_features), didapat {centers_arr.shape}")

    n_samples, n_features = X_arr.shape
    c, c_features = centers_arr.shape

    if n_features != c_features:
        raise ValueError(
            f"Dimensi fitur data X ({n_features}) tidak cocok dengan pusat cluster ({c_features})."
        )

    # Input skfuzzy: (n_features, n_samples)
    test_data = X_arr.T

    if SKFUZZY_AVAILABLE:
        try:
            u, u0, d, jm, p, fpc = skf_cluster.cmeans_predict(
                test_data=test_data,
                cntr_trained=centers_arr,
                m=float(m),
                error=float(error),
                maxiter=int(maxiter),
                metric="euclidean",
                seed=int(seed),
            )
            # Transpose kembali ke (n_samples, c)
            return u.T
        except Exception:
            pass

    # Fallback NumPy murni
    u, u0, d, jm, p, fpc = _cmeans_predict_numpy(
        test_data=test_data,
        cntr_trained=centers_arr,
        m=float(m),
        error=float(error),
        maxiter=int(maxiter),
        metric="euclidean",
        seed=int(seed),
    )
    return u.T


# ==============================================================================
# Class FCM Wrapper scikit-fuzzy
# ==============================================================================

class FCM:
    """
    Wrapper Library scikit-fuzzy untuk Fuzzy C-Means (cmeans & cmeans_predict).

    Parameters
    ----------
    n_clusters : int, default=6
        Jumlah cluster (c).
    m : float, default=2.0
        Fuzziness exponent (m > 1.0).
    max_iter : int, default=200
        Batas maksimum jumlah iterasi (maxiter).
    error : float, default=1e-5
        Toleransi ambang konvergensi (xi / epsilon).
    random_state : int, default=42
        Seed acak untuk reproduktibilitas inisialisasi u0.

    Attributes
    ----------
    centers_ : np.ndarray of shape (c, n_features)
        Pusat cluster V hasil konvergensi cmeans.
    u_ : np.ndarray of shape (n_samples, c)
        Matriks keanggotaan akhir pada data latih (sudah ditranspose dari format skfuzzy).
    objective_history_ : list of float
        Riwayat nilai fungsi objektif J (jm) pada setiap iterasi.
    final_objective_ : float
        Nilai fungsi objektif akhir P (jm[-1]).
    fpc_ : float
        Fuzzy Partition Coefficient (FPC).
    pe_ : float
        Partition Entropy (PE).
    silhouette_ : float or None
        Silhouette score pada ruang data latih.
    n_iter_ : int
        Jumlah iterasi hingga konvergen (p).
    labels_ : np.ndarray of shape (n_samples,)
        Label cluster crisp (argmax matriks u_).
    """

    def __init__(
        self,
        n_clusters: int = 6,
        m: float = 2.0,
        max_iter: int = 200,
        error: float = 1e-5,
        random_state: int = 42,
    ):
        if n_clusters < 1:
            raise ValueError(f"n_clusters harus >= 1, didapat {n_clusters}")
        if m <= 1.0:
            raise ValueError(f"m harus > 1.0, didapat {m}")
        if max_iter < 1:
            raise ValueError(f"max_iter harus >= 1, didapat {max_iter}")
        if error <= 0.0:
            raise ValueError(f"error harus > 0.0, didapat {error}")

        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.error = float(error)
        self.random_state = int(random_state)

        # Atribut hasil fit
        self.centers_: Optional[np.ndarray] = None
        self.u_: Optional[np.ndarray] = None
        self.objective_history_: List[float] = []
        self.final_objective_: Optional[float] = None
        self.fpc_: Optional[float] = None
        self.pe_: Optional[float] = None
        self.silhouette_: Optional[float] = None
        self.n_iter_: int = 0
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FCM":
        """
        Melatih Fuzzy C-Means pada matriks data X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data masukan (misal fitur hasil reduksi dimensi PCA).
            
        Returns
        -------
        self : FCM
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X harus berupa matriks 2D (n_samples, n_features), didapat {X_arr.shape}")

        n_samples, n_features = X_arr.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                f"Jumlah sampel ({n_samples}) tidak boleh lebih kecil dari jumlah cluster ({self.n_clusters})"
            )

        # skfuzzy menerima data dalam format (n_features, n_samples)
        data_transposed = X_arr.T

        if SKFUZZY_AVAILABLE:
            try:
                cntr, u, u0, d, jm, p, fpc = skf_cluster.cmeans(
                    data=data_transposed,
                    c=self.n_clusters,
                    m=self.m,
                    error=self.error,
                    maxiter=self.max_iter,
                    metric="euclidean",
                    seed=self.random_state,
                )
            except Exception:
                cntr, u, u0, d, jm, p, fpc = _cmeans_numpy(
                    data=data_transposed,
                    c=self.n_clusters,
                    m=self.m,
                    error=self.error,
                    maxiter=self.max_iter,
                    metric="euclidean",
                    seed=self.random_state,
                )
        else:
            cntr, u, u0, d, jm, p, fpc = _cmeans_numpy(
                data=data_transposed,
                c=self.n_clusters,
                m=self.m,
                error=self.error,
                maxiter=self.max_iter,
                metric="euclidean",
                seed=self.random_state,
            )

        # Transposisikan u dari (c, n_samples) ke (n_samples, c)
        u_standard = u.T

        self.centers_ = cntr
        self.u_ = u_standard
        self.objective_history_ = list(jm)
        self.final_objective_ = float(jm[-1]) if len(jm) > 0 else 0.0
        self.n_iter_ = int(p)
        self.fpc_ = float(fpc)
        self.pe_ = compute_partition_entropy(u_standard)
        self.labels_ = np.argmax(u_standard, axis=1)
        self.silhouette_ = compute_fcm_silhouette(X_arr, u_standard)

        return self

    def compute_membership(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung matriks derajat keanggotaan mu untuk data baru terhadap pusat cluster.
        Mengembalikan matriks berbentuk (n_samples, c).
        """
        if self.centers_ is None:
            raise RuntimeError("Model FCM belum dilatih. Panggil method fit() terlebih dahulu.")

        return fcm_predict_membership(
            X,
            self.centers_,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            seed=self.random_state,
        )

    def predict_membership(self, X: np.ndarray) -> np.ndarray:
        """Alias untuk compute_membership(X)."""
        return self.compute_membership(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Memprediksi label cluster crisp (argmax mu) untuk data baru."""
        mu = self.compute_membership(X)
        return np.argmax(mu, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Melatih model dan mengembalikan label cluster crisp data latih."""
        self.fit(X)
        return self.labels_
