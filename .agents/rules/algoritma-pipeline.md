---
trigger: glob
globs: "src/**/*.py, notebooks/**/*.ipynb, config.yaml, app.py"
description: "Aturan ekstraksi fitur MobileNetV2/Handcrafted, PCA, FCM manual dengan PE, KNN dengan mu, pipeline training/prediksi, dan penyimpanan model."
---

# Algoritma dan pipeline

Pakai spesifikasi di @SPESIFIKASI-PROYEK.md sebagai acuan angka fitur dan parameter. Bagian ini merinci kontrak kode.

## features.py

- Input: path gambar atau array RGB/BGR/PIL Image. Validasi format dan handle file rusak dengan error yang jelas.
- Ekstraktor Fitur:
  1. **MobileNetV2 (Utama):** Pretrained ImageNet (`include_top=False, pooling='avg'`), input citra 224x224, diproses menggunakan `tf.keras.applications.mobilenet_v2.preprocess_input`. Menghasilkan 1280 fitur float. Model di-cache agar efisien. Fallback gracefully jika tensorflow belum terpasang atau mode handcrafted diminta.
  2. **Handcrafted (Pembanding):**
     - Histogram HSV: 8 bin H + 4 bin S + 4 bin V dinormalisasi sum=1 (16 angka).
     - Statistik warna: Mean dan standard deviation tiap kanal H, S, V (6 angka).
     - Tekstur GLCM pada grayscale: Contrast, dissimilarity/homogeneity, energy, correlation (4 angka).
     - Canny edge density opsional jika config aktif.
     - Total: ~26 dimensi.
- Fungsi: `extract_features(image_or_path, feature_extractor="mobilenetv2", config=None, model=None) -> np.ndarray`.

## fcm.py (manual, NumPy)

Implementasikan FCM sendiri menggunakan NumPy murni tanpa pustaka scikit-fuzzy eksternal.

- Parameter: `c`, `m=2.0`, `max_iter`, `epsilon`, `random_state`.
- Inisialisasi matriks keanggotaan `U` acak dinormalisasi sehingga $\sum_j u_{ij} = 1$ per sampel.
- Pusat cluster: $v_j = \sum_i u_{ij}^m x_i / \sum_i u_{ij}^m$.
- Jarak Euclidean $d_{ij} = \|x_i - v_j\|$. Lindungi jarak 0 dengan nilai epsilon kecil agar tidak divide-by-zero.
- Pembaruan keanggotaan: $u_{ij} = 1 / \sum_k (d_{ij}/d_{ik})^{2/(m-1)}$.
- Fungsi objektif: $J = \sum_i \sum_j u_{ij}^m d_{ij}^2$. Catat riwayat per iterasi di `objective_history_`.
- Evaluasi partisi fuzzy:
  * Fuzzy Partition Coefficient: $\text{FPC} = \frac{1}{n} \sum_i \sum_j u_{ij}^2$.
  * Partition Entropy: $\text{PE} = -\frac{1}{n} \sum_i \sum_j u_{ij} \ln(u_{ij})$ (untuk $u_{ij} > 0$).
- Atribut class FCM: `centers_`, `u_`, `objective_history_`, `fpc_`, `pe_`, `n_iter_`.
- Fungsi `fcm_predict_membership(X, centers, m=2.0) -> np.ndarray`: Menghitung keanggotaan sampel baru/data uji berdasarkan jarak ke pusat cluster tanpa melakukan iterasi training ulang.

## pipeline.py

Satu pintu untuk notebook Colab dan Streamlit:

- **Pemisahan Data (Anti Data Leakage):**
  Lakukan stratified train/test split (`test_size=0.2`, `random_state=42`) **SEBELUM** melakukan fitting.
- `StandardScaler` di-fit **HANYA** pada data train.
- `PCA` di-fit **HANYA** pada data train yang telah diskalakan (`variance_threshold=0.90`, dibatasi `max_components=50`).
- `FCM` di-fit **HANYA** pada data train di ruang PCA.
- Derajat keanggotaan data uji dihitung melalui `fcm_predict_membership(X_test_pca, centers, m)`.
- Input KNN: Gabungan fitur PCA dan derajat keanggotaan fuzzy dikalikan bobot:
  $$\mathbf{X}_{KNN} = [\mathbf{X}_{PCA}, \boldsymbol{\mu} \times \text{mu\_weight}]$$
  Dilatih menggunakan label kelas aktual.
- Fungsi perbandingan:
  * `compare_knn_variants(...)`: KNN biasa (hanya fitur PCA) vs KNN + mu FCM (fitur PCA + mu).
  * `compare_feature_extractors(...)`: MobileNetV2 vs Handcrafted HSV+GLCM.
- `predict(image_or_path, models, config)`: ekstraksi -> scaler -> PCA -> hitung mu via `fcm_predict_membership` -> gabung `[PCA, mu * mu_weight]` -> KNN predict.
- `save_models(models_dir)` & `load_models(models_dir)`: Mengelola `scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, dan `meta.json`. Latih ulang dari GUI menyimpan ke `st.session_state`, tidak otomatis menimpa file di disk.
