"""
Modul Fuzzy C-Means (FCM) Manual NumPy (fcm.py)
==============================================
Implementasi manual algoritma clustering Fuzzy C-Means (FCM) murni dengan NumPy (tanpa scikit-fuzzy).
Fitur & Spesifikasi:
1. Perhitungan pusat cluster V:
     v_j = sum_i(u_ij^m * x_i) / sum_i(u_ij^m)
2. Pembaruan matriks keanggotaan U dengan proteksi divide-by-zero (jika d_ij = 0).
3. Evaluasi fungsi objektif J dan pencatatan riwayat J per iterasi.
4. Metrik Unsupervised:
   - Fuzzy Partition Coefficient (FPC):
       FPC = (1 / n) * sum_i sum_j (u_ij^2)  [rentang: 1/c s.d. 1.0]
   - Partition Entropy (PE):
       PE = - (1 / n) * sum_i sum_j (u_ij * ln(u_ij)) untuk u_ij > 0  [rentang: 0 s.d. ln(c)]
5. Fungsi mandiri `fcm_predict_membership(X, centers, m=2.0)` untuk menghitung keanggotaan data uji/baru
   terhadap pusat cluster yang sudah dilatih TANPA menjalankan FCM ulang.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def fcm_predict_membership(
    X: np.ndarray,
    centers: np.ndarray,
    m: float = 2.0,
) -> np.ndarray:
    """
    Menghitung derajat keanggotaan mu_ij untuk data baru / test berdasarkan
    jarak ke pusat cluster V hasil training (BUKAN menjalankan FCM ulang!).

    Formula:
        mu_ij = 1 / sum_k( (d_ij / d_ik)^(2 / (m - 1)) )
              = (d_ij)^(-p) / sum_k( (d_ik)^(-p) ), dengan p = 2 / (m - 1)

    Dilengkapi penanganan kondisi khusus jika d_ij == 0 (sampel tepat di pusat cluster).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features) atau 1D (n_features,)
        Data uji atau data baru.
    centers : np.ndarray of shape (n_clusters, n_features)
        Pusat cluster V hasil training FCM.
    m : float, default=2.0
        Fuzziness exponent (m > 1.0).

    Returns
    -------
    np.ndarray of shape (n_samples, n_clusters)
        Matriks derajat keanggotaan mu_ij di mana sum_j(mu_ij) = 1.0 untuk tiap sampel.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if centers_arr.ndim != 2:
        raise ValueError(f"centers harus berupa array 2D (n_clusters, n_features), didapat {centers_arr.shape}")

    n_samples, n_features = X_arr.shape
    n_clusters, center_features = centers_arr.shape

    if n_features != center_features:
        raise ValueError(
            f"Dimensi fitur data X ({n_features}) tidak cocok dengan pusat cluster ({center_features})."
        )
    if m <= 1.0:
        raise ValueError(f"Fuzziness exponent m harus > 1.0, didapat {m}")

    # 1. Hitung jarak Euclidean pairwise antara data X dan setiap pusat cluster
    # diff: (n_samples, n_clusters, n_features)
    diff = X_arr[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)  # (n_samples, n_clusters)

    p = 2.0 / (m - 1.0)
    mu = np.zeros((n_samples, n_clusters), dtype=np.float64)

    # Cek perlindungan divide-by-zero (ketika jarak ke salah satu pusat = 0)
    zero_mask = dist < 1e-12
    rows_with_zeros = zero_mask.any(axis=1)

    # A. Kasus normal: sampel tidak berimpit dengan pusat manapun
    normal_rows = ~rows_with_zeros
    if np.any(normal_rows):
        d_norm = dist[normal_rows]
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist_p = d_norm ** (-p)
        sum_inv = inv_dist_p.sum(axis=1, keepdims=True)
        sum_inv = np.where(sum_inv == 0, 1e-12, sum_inv)
        mu[normal_rows] = inv_dist_p / sum_inv

    # B. Kasus khusus: sampel tepat berada di salah satu pusat cluster (dist = 0)
    if np.any(rows_with_zeros):
        for i in np.where(rows_with_zeros)[0]:
            zero_cols = np.where(zero_mask[i])[0]
            num_zeros = len(zero_cols)
            mu[i, :] = 0.0
            mu[i, zero_cols] = 1.0 / num_zeros

    return mu


def compute_fpc(u: np.ndarray) -> float:
    """
    Menghitung Fuzzy Partition Coefficient (FPC).
    FPC = (1 / n) * sum_i sum_j (u_ij^2)
    """
    n = u.shape[0]
    if n == 0:
        return 0.0
    return float(np.sum(u ** 2) / n)


def compute_partition_entropy(u: np.ndarray) -> float:
    """
    Menghitung Partition Entropy (PE).
    PE = - (1 / n) * sum_i sum_j (u_ij * ln(u_ij)) untuk u_ij > 0.
    Semakin kecil nilai PE, semakin tegas (crisp) dan terstruktur partisinya.
    """
    n = u.shape[0]
    if n == 0:
        return 0.0
    # Mask untuk nilai positif di atas toleransi numerik
    positive_mask = u > 1e-12
    val = np.zeros_like(u)
    val[positive_mask] = u[positive_mask] * np.log(u[positive_mask])
    pe = - float(np.sum(val) / n)
    return max(0.0, pe)


# Alias untuk kemudahan import
compute_pe = compute_partition_entropy


class FCM:
    """
    Fuzzy C-Means Clustering (NumPy murni).

    Parameters
    ----------
    n_clusters : int, default=6
        Jumlah cluster (c).
    m : float, default=2.0
        Fuzziness exponent (m > 1.0).
    max_iter : int, default=150
        Batas maksimum jumlah iterasi.
    error : float, default=1e-5
        Toleransi konvergensi (epsilon).
    random_state : int or None, default=42
        Seed untuk reproduktibilitas inisialisasi U.

    Attributes
    ----------
    centers_ : np.ndarray of shape (n_clusters, n_features)
        Pusat cluster V.
    u_ : np.ndarray of shape (n_samples, n_clusters)
        Matriks derajat keanggotaan akhir pada data latih.
    objective_history_ : list of float
        Riwayat nilai fungsi objektif J pada tiap iterasi.
    fpc_ : float
        Fuzzy Partition Coefficient (FPC).
    pe_ : float
        Partition Entropy (PE).
    n_iter_ : int
        Jumlah iterasi hingga konvergensi.
    labels_ : np.ndarray of shape (n_samples,)
        Label cluster crisp hasil argmax matriks keanggotaan U.
    """

    def __init__(
        self,
        n_clusters: int = 6,
        m: float = 2.0,
        max_iter: int = 150,
        error: float = 1e-5,
        random_state: Optional[int] = 42,
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
        self.random_state = random_state

        # Atribut hasil fit
        self.centers_: Optional[np.ndarray] = None
        self.u_: Optional[np.ndarray] = None
        self.objective_history_: List[float] = []
        self.fpc_: Optional[float] = None
        self.pe_: Optional[float] = None
        self.n_iter_: int = 0
        self.labels_: Optional[np.ndarray] = None

    def _init_u(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        """Inisialisasi matriks keanggotaan U secara acak dan dinormalisasi sum=1 per baris."""
        u_init = rng.uniform(0.01, 1.0, size=(n_samples, self.n_clusters))
        row_sums = u_init.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return u_init / row_sums

    def _compute_centers(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Menghitung pusat cluster v_j berdasarkan data X dan matriks U.
        v_j = sum_i(u_ij^m * x_i) / sum_i(u_ij^m)
        """
        u_m = u ** self.m
        numerator = np.dot(u_m.T, X)
        denominator = u_m.sum(axis=0, keepdims=True).T
        denominator = np.where(denominator == 0, 1e-12, denominator)
        return numerator / denominator

    def _compute_objective(self, X: np.ndarray, centers: np.ndarray, u: np.ndarray) -> float:
        """Menghitung nilai fungsi objektif J = sum_i sum_j (u_ij^m * ||x_i - v_j||^2)."""
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        u_m = u ** self.m
        return float(np.sum(u_m * dist_sq))

    def fit(self, X: np.ndarray) -> "FCM":
        """
        Melatih model FCM pada data masukan X (misal fitur hasil PCA).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X harus berupa matriks 2D (n_samples, n_features), didapat {X.shape}")

        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                f"Jumlah sampel ({n_samples}) tidak boleh lebih kecil dari jumlah cluster ({self.n_clusters})"
            )

        rng = np.random.RandomState(self.random_state)
        u = self._init_u(n_samples, rng)
        self.objective_history_ = []

        for it in range(self.max_iter):
            # 1. Hitung pusat cluster
            centers = self._compute_centers(X, u)

            # 2. Hitung fungsi objektif J
            j_val = self._compute_objective(X, centers, u)
            self.objective_history_.append(j_val)

            # 3. Update keanggotaan menggunakan rumus membership
            u_new = fcm_predict_membership(X, centers, m=self.m)

            # 4. Cek konvergensi
            diff_u = float(np.max(np.abs(u_new - u)))
            diff_j = (
                float(abs(self.objective_history_[-1] - self.objective_history_[-2]))
                if len(self.objective_history_) > 1
                else float("inf")
            )

            u = u_new

            if diff_u < self.error or diff_j < self.error:
                break

        # Simpan state akhir model
        self.centers_ = centers
        self.u_ = u
        self.fpc_ = compute_fpc(u)
        self.pe_ = compute_partition_entropy(u)
        self.n_iter_ = len(self.objective_history_)
        self.labels_ = np.argmax(u, axis=1)

        return self

    def compute_membership(self, X: np.ndarray) -> np.ndarray:
        """
        Menghitung matriks derajat keanggotaan mu untuk data baru terhadap pusat cluster FCM.
        """
        if self.centers_ is None:
            raise RuntimeError("Model FCM belum dilatih. Panggil method fit() terlebih dahulu.")
        return fcm_predict_membership(X, self.centers_, m=self.m)

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
