"""
Modul Ekstraksi Fitur Citra Sampah (features.py)
================================================
Mendukung dua jalur ekstraksi fitur:
1. MobileNetV3-Small (Transfer Learning Keras / TensorFlow) - Jalur Utama (Deep Features):
   - Arsitektur: tf.keras.applications.MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg", include_preprocessing=True)
   - Input: citra RGB resolusi 224x224 dengan nilai piksel float [0, 255]
     (Catatan: JANGAN normalisasi atau bagi 255 di luar model karena include_preprocessing=True sudah otomatis
      menangani rescaling dan normalisasi bawaan MobileNetV3 di dalam graf komputasi model).
   - Vektor output: 576 dimensi float (representasi fitur visual kompak dan kaya informasi semantik).
   - Menggunakan mekanisme caching model dalam memori untuk kecepatan inferensi dan efisiensi memori.
2. Handcrafted Features (Fitur Buatan Tangan) - Jalur Pembanding:
   - 16 fitur Histogram HSV (8 bin H, 4 bin S, 4 bin V, sum=1).
   - 6 fitur Statistik Warna (mean & std kanal H, S, V).
   - 4 fitur Tekstur GLCM (Contrast, Homogeneity, Energy, Correlation).
   - Opsional 1 Canny Edge Density.
   - Total: ~26 dimensi (27 jika use_edge_density=True).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Cek ketersediaan TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Cek ketersediaan scikit-image untuk GLCM
try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Cek ketersediaan OpenCV untuk Canny
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Cache model MobileNetV3-Small dalam memori
_CACHED_MOBILENET_MODEL = None


def get_mobilenet_model() -> Any:
    """
    Memuat dan meng-cache instance model MobileNetV3Small dengan bobot ImageNet.
    
    Arsitektur:
        tf.keras.applications.MobileNetV3Small(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            include_preprocessing=True,
            input_shape=(224, 224, 3)
        )
    
    Returns
    -------
    tf.keras.Model
        Model ekstraktor fitur MobileNetV3-Small dengan output 576 dimensi.
        
    Raises
    ------
    ImportError
        Jika TensorFlow belum terpasang di sistem.
    RuntimeError
        Jika gagal mengunduh bobot ImageNet saat inisialisasi awal.
    """
    global _CACHED_MOBILENET_MODEL
    if _CACHED_MOBILENET_MODEL is not None:
        return _CACHED_MOBILENET_MODEL

    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow tidak terdeteksi di environment ini. "
            "Untuk menggunakan ekstraktor fitur MobileNetV3-Small (576 dimensi), "
            "silakan instal TensorFlow (`pip install tensorflow`). "
            "Sebagai alternatif, gunakan `feature_extractor='handcrafted'`."
        )

    try:
        model = tf.keras.applications.MobileNetV3Small(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            include_preprocessing=True,
            input_shape=(224, 224, 3),
        )
        _CACHED_MOBILENET_MODEL = model
        return _CACHED_MOBILENET_MODEL
    except Exception as e:
        raise RuntimeError(
            f"Gagal menginisialisasi model MobileNetV3Small dengan bobot ImageNet: {e}. "
            "Pastikan koneksi internet aktif untuk download bobot pertama kali, "
            "atau pilih mode feature_extractor='handcrafted'."
        )


def load_image(
    image_or_path: Union[str, Path, Image.Image, np.ndarray],
    target_size: Optional[Tuple[int, int]] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memuat dan memvalidasi citra input menjadi format RGB dan Grayscale dengan resolusi standar.
    
    Parameters
    ----------
    image_or_path : str, Path, Image.Image, atau np.ndarray
        Filepath gambar, objek PIL Image, atau array NumPy.
    target_size : tuple of int, optional
        Ukuran resize (width, height), default (224, 224).
        
    Returns
    -------
    tuple of (rgb_uint8, gray_uint8)
        rgb_uint8 : np.ndarray berukuran (H, W, 3) bernilai [0, 255].
        gray_uint8 : np.ndarray berukuran (H, W) bernilai [0, 255].
    """
    img_pil: Optional[Image.Image] = None

    if isinstance(image_or_path, (str, Path)):
        p = Path(image_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"File citra tidak ditemukan: {image_or_path}")
        try:
            with Image.open(p) as img:
                img_pil = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Gagal membaca file gambar {image_or_path}: {e}")

    elif isinstance(image_or_path, Image.Image):
        img_pil = image_or_path.convert("RGB")

    elif isinstance(image_or_path, np.ndarray):
        arr = image_or_path
        if arr.size == 0:
            raise ValueError("Array citra kosong (size == 0).")
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            img_pil = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                img_pil = Image.fromarray(arr[:, :, 0]).convert("RGB")
            elif arr.shape[2] == 3:
                img_pil = Image.fromarray(arr)
            elif arr.shape[2] == 4:
                img_pil = Image.fromarray(arr).convert("RGB")
            else:
                raise ValueError(f"Jumlah kanal citra tidak didukung: {arr.shape}")
        else:
            raise ValueError(f"Dimensi array tidak valid: {arr.ndim}")
    else:
        raise TypeError(f"Tipe data {type(image_or_path)} tidak didukung sebagai citra.")

    if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
        img_pil = img_pil.resize(target_size, Image.Resampling.BILINEAR)

    rgb = np.array(img_pil, dtype=np.uint8)
    gray = np.dot(rgb[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140]).clip(0, 255).astype(np.uint8)
    return rgb, gray


def extract_mobilenet_features(
    image_or_path: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[Any] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Mengekstrak 576 fitur representasi visual menggunakan MobileNetV3-Small pretrained ImageNet.
    
    PENTING:
    - Input citra RGB di-resize ke 224x224 dengan nilai piksel float [0, 255].
    - JANGAN rescale/bagi 255 di luar model karena include_preprocessing=True sudah bawaan Keras.
    
    Parameters
    ----------
    image_or_path : str, Path, Image.Image, atau np.ndarray
    model : tf.keras.Model, optional
        Model MobileNetV3Small yang sudah di-cache (jika None, memanggil get_mobilenet_model()).
    target_size : tuple of int, default (224, 224)
        
    Returns
    -------
    np.ndarray
        Array 1D berukuran (576,) float64.
    """
    rgb, _ = load_image(image_or_path, target_size=target_size)
    # Nilai piksel dalam format float32 rentang [0, 255]
    x = rgb.astype(np.float32)
    x_batch = np.expand_dims(x, axis=0)

    if model is None:
        model = get_mobilenet_model()

    features = model.predict(x_batch, verbose=0)
    features_1d = np.squeeze(features).astype(np.float64)
    return features_1d


# ==============================================================================
# Helper Ekstraksi Fitur Handcrafted (HSV + GLCM)
# ==============================================================================

def _rgb_to_hsv_numpy(rgb: np.ndarray) -> np.ndarray:
    """Konversi citra RGB ternormalisasi [0, 1] ke HSV [0, 1] secara murni dengan NumPy."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc.copy()
    deltac = maxc - minc

    s = np.zeros_like(v)
    nonzero_mask = maxc > 1e-7
    s[nonzero_mask] = deltac[nonzero_mask] / maxc[nonzero_mask]

    h = np.zeros_like(v)
    delta_mask = deltac > 1e-7

    mask_r = delta_mask & (maxc == r)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / deltac[mask_r]) % 6.0

    mask_g = delta_mask & (maxc == g) & ~mask_r
    h[mask_g] = ((b[mask_g] - r[mask_g]) / deltac[mask_g]) + 2.0

    mask_b = delta_mask & (maxc == b) & ~mask_r & ~mask_g
    h[mask_b] = ((r[mask_b] - g[mask_b]) / deltac[mask_b]) + 4.0

    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def _compute_glcm_features(gray: np.ndarray) -> np.ndarray:
    """Menghitung 4 fitur GLCM (Contrast, Homogeneity, Energy, Correlation)."""
    if SKIMAGE_AVAILABLE:
        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = float(graycoprops(glcm, 'contrast')[0, 0])
            homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
            energy = float(graycoprops(glcm, 'energy')[0, 0])
            correlation = float(graycoprops(glcm, 'correlation')[0, 0])
            if np.isnan(correlation):
                correlation = 0.0
            return np.array([contrast, homogeneity, energy, correlation], dtype=np.float64)
        except Exception:
            pass

    # Fallback NumPy murni: Kuantisasi ke 32 level
    levels = 32
    gray_q = (gray.astype(np.float64) / 256.0 * levels).astype(np.int32)
    gray_q = np.clip(gray_q, 0, levels - 1)

    p = np.zeros((levels, levels), dtype=np.float64)
    left = gray_q[:, :-1].ravel()
    right = gray_q[:, 1:].ravel()
    np.add.at(p, (left, right), 1.0)
    p = p + p.T
    total = np.sum(p)
    p = p / total if total > 0 else np.ones((levels, levels), dtype=np.float64) / (levels * levels)

    i_indices, j_indices = np.indices((levels, levels), dtype=np.float64)
    diff = i_indices - j_indices

    contrast = float(np.sum((diff ** 2) * p))
    homogeneity = float(np.sum(p / (1.0 + diff ** 2)))
    energy = float(np.sum(p ** 2))

    mean_i = np.sum(i_indices * p)
    mean_j = np.sum(j_indices * p)
    var_i = np.sum(((i_indices - mean_i) ** 2) * p)
    var_j = np.sum(((j_indices - mean_j) ** 2) * p)
    std_prod = np.sqrt(var_i * var_j)
    correlation = float(np.sum((i_indices - mean_i) * (j_indices - mean_j) * p) / std_prod) if std_prod > 1e-9 else 0.0

    return np.array([contrast, homogeneity, energy, correlation], dtype=np.float64)


def _compute_edge_density(gray: np.ndarray) -> float:
    """Menghitung kepadatan tepi (Canny edge density)."""
    if CV2_AVAILABLE:
        try:
            edges = cv2.Canny(gray, 100, 200)
            return float(np.mean(edges > 0))
        except Exception:
            pass

    gy, gx = np.gradient(gray.astype(np.float64))
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(magnitude > 30.0))


def extract_handcrafted_features(
    image_or_path: Union[str, Path, Image.Image, np.ndarray],
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Mengekstrak fitur buatan tangan (~26 dimensi) dari foto sampah:
    - 16 Histogram HSV (8 bin H, 4 bin S, 4 bin V)
    - 6 Statistik Warna (mean, std kanal H, S, V)
    - 4 Tekstur GLCM (Contrast, Homogeneity, Energy, Correlation)
    - [Opsional] 1 Canny edge density jika use_edge_density=True
    
    Returns
    -------
    np.ndarray
        Array 1D float64 (26 atau 27 dimensi).
    """
    target_size = (128, 128)
    use_edge_density = False

    if config and isinstance(config, dict):
        feat_cfg = config.get("features", {})
        if isinstance(feat_cfg, dict):
            use_edge_density = feat_cfg.get("use_edge_density", False)
            ts = feat_cfg.get("handcrafted_size", feat_cfg.get("target_size", [128, 128]))
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                target_size = (int(ts[0]), int(ts[1]))

    rgb, gray = load_image(image_or_path, target_size=target_size)
    rgb_float = rgb.astype(np.float64) / 255.0
    hsv = _rgb_to_hsv_numpy(rgb_float)

    h_channel = hsv[..., 0].ravel()
    s_channel = hsv[..., 1].ravel()
    v_channel = hsv[..., 2].ravel()

    # 1. Histogram HSV (16 fitur)
    hist_h, _ = np.histogram(h_channel, bins=8, range=(0.0, 1.0))
    hist_s, _ = np.histogram(s_channel, bins=4, range=(0.0, 1.0))
    hist_v, _ = np.histogram(v_channel, bins=4, range=(0.0, 1.0))

    norm_h = hist_h.astype(np.float64) / (np.sum(hist_h) if np.sum(hist_h) > 0 else 1.0)
    norm_s = hist_s.astype(np.float64) / (np.sum(hist_s) if np.sum(hist_s) > 0 else 1.0)
    norm_v = hist_v.astype(np.float64) / (np.sum(hist_v) if np.sum(hist_v) > 0 else 1.0)
    hsv_hist = np.concatenate([norm_h, norm_s, norm_v])

    # 2. Statistik Warna (6 fitur)
    color_stats = np.array([
        np.mean(h_channel), np.std(h_channel),
        np.mean(s_channel), np.std(s_channel),
        np.mean(v_channel), np.std(v_channel),
    ], dtype=np.float64)

    # 3. Tekstur GLCM (4 fitur)
    glcm_feat = _compute_glcm_features(gray)

    feature_list = [hsv_hist, color_stats, glcm_feat]

    # 4. Opsional: Edge Density (1 fitur)
    if use_edge_density:
        ed = _compute_edge_density(gray)
        feature_list.append(np.array([ed], dtype=np.float64))

    return np.concatenate(feature_list).astype(np.float64)


# ==============================================================================
# Fungsi Utama Ekstraksi Fitur
# ==============================================================================

def extract_features(
    image_or_path: Union[str, Path, Image.Image, np.ndarray],
    feature_extractor: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
) -> np.ndarray:
    """
    Ekstraksi fitur citra sampah dengan dukungan MobileNetV3-Small (576D) atau Handcrafted (~26D).
    
    Parameters
    ----------
    image_or_path : str, Path, Image.Image, atau np.ndarray
    feature_extractor : str, optional
        Pilihan mode ekstraktor: "mobilenetv3_small" (576 dimensi) atau "handcrafted" (~26 dimensi).
        Jika None, diambil dari config.features.feature_extractor (default "mobilenetv3_small").
    config : dict, optional
        Konfigurasi parameter.
    model : tf.keras.Model, optional
        Instance model MobileNetV3-Small yang telah di-cache.
        
    Returns
    -------
    np.ndarray
        Array 1D float64 hasil ekstraksi fitur.
    """
    if feature_extractor is None:
        if config and isinstance(config, dict):
            feat_cfg = config.get("features", {})
            if isinstance(feat_cfg, dict):
                feature_extractor = feat_cfg.get("feature_extractor", "mobilenetv3_small")
        else:
            feature_extractor = "mobilenetv3_small"

    extractor_mode = str(feature_extractor).lower()

    if extractor_mode in ["mobilenetv3_small", "mobilenetv3", "mobilenet_v3_small", "mobilenet", "deep_learning"]:
        target_size = (224, 224)
        if config and isinstance(config, dict):
            feat_cfg = config.get("features", {})
            if isinstance(feat_cfg, dict):
                ts = feat_cfg.get("image_size", [224, 224])
                if isinstance(ts, (list, tuple)) and len(ts) == 2:
                    target_size = (int(ts[0]), int(ts[1]))

        try:
            return extract_mobilenet_features(image_or_path, model=model, target_size=target_size)
        except Exception as e:
            if config and isinstance(config, dict) and config.get("features", {}).get("allow_handcrafted_fallback", False):
                return extract_handcrafted_features(image_or_path, config=config)
            raise e

    elif extractor_mode in ["handcrafted", "hsv_glcm", "manual"]:
        return extract_handcrafted_features(image_or_path, config=config)

    else:
        raise ValueError(
            f"Mode ekstraktor '{feature_extractor}' tidak dikenal. "
            "Pilih 'mobilenetv3_small' atau 'handcrafted'."
        )


def get_feature_names(
    feature_extractor: str = "mobilenetv3_small",
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Mengembalikan daftar nama fitur berdasarkan mode ekstraktor yang dipilih.
    """
    mode = str(feature_extractor).lower()
    if mode in ["mobilenetv3_small", "mobilenetv3", "mobilenet_v3_small", "mobilenet", "deep_learning"]:
        return [f"mobilenetv3_feat_{i}" for i in range(576)]

    names: List[str] = []
    for i in range(8):
        names.append(f"hsv_hist_h_bin{i}")
    for i in range(4):
        names.append(f"hsv_hist_s_bin{i}")
    for i in range(4):
        names.append(f"hsv_hist_v_bin{i}")

    names.extend(["h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std"])
    names.extend(["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"])

    if config and config.get("features", {}).get("use_edge_density", False):
        names.append("edge_density")

    return names
