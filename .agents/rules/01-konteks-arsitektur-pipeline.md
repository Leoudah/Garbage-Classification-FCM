---
trigger: glob
globs: "src/**/*.py, notebooks/**/*.ipynb, config.yaml, app.py"
description: "Konteks sistem, arsitektur full library, dataset, rumus internal FCM+KNN, parameter resmi, dan contoh manual Kilap/Coklat."
---

# Konteks arsitektur pipeline (tahap implementasi)

Dokumen ini merangkum **Dokumen Konteks Sistem** dan **Contoh Perhitungan Manual** Kelompok 5, disesuaikan dengan keputusan resmi: implementasi memakai **full library**, sementara rumus internal tetap wajib dipahami.

Jangan merancang ulang arsitektur. Lanjutkan dari kode `staging` (commit `367fb93`), lalu rapikan celah yang masih ada.

## 1. Framing masalah dan jenis sistem

Pemilahan sampah di TPS / Bank Sampah masih rentan kekeliruan, terutama pada benda yang tampak mirip (plastik bening vs kaca, kardus vs kertas). Keputusan “ya atau tidak” terasa kaku, karena satu benda bisa mirip lebih dari satu kelompok.

Sistem bersifat **hibrida**:

- **FCM tak terawasi (unsupervised).** Label kelas tidak dipakai. FCM hanya mencari pola kemiripan di ruang fitur PCA dan menghasilkan pusat cluster `V` serta matriks derajat keanggotaan `U` / `μ`.
- **KNN terawasi (supervised).** KNN memakai label asli dataset untuk menentukan kelas.
- **Integrasi nyata** terjadi pada masukan KNN:

```text
X_KNN = [X_PCA  ||  alpha * mu]
```

Tanpa keluaran FCM, vektor masukan KNN pada rancangan ini tidak lengkap. Satu cluster FCM **tidak otomatis** sama dengan satu kelas sampah. Jumlah `c = 6` hanya diselaraskan dengan jumlah kelas; nama kelas baru muncul di tahap KNN.

Enam kelas resmi: `plastic`, `metal`, `glass`, `cardboard`, `paper`, `trash`.

## 2. Dataset, sampling, dan split (anti leakage)

- Sumber: [Garbage Images Dataset](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification), slug `zlatan599/garbage-dataset-classification`, lisensi MIT.
- Citra RGB, ukuran asli sekitar 256×256, sudah deduplikasi.
- **Sampling resmi:** 1.000 gambar acak per kelas, `seed=42`, total 6.000 gambar. Wajib acak (bukan `img_paths[:1000]` atau 100/30 sampel “sementara” pada training resmi).
- **Split resmi:** stratified 80:20, `seed=42` → 4.800 latih (800/kelas) dan 1.200 uji (200/kelas).
- Split dilakukan **paling awal**, sebelum ekstraksi yang di-fit, `StandardScaler`, PCA, dan FCM.
- Fungsi yang sudah benar di `staging` dan wajib dipakai notebook: `load_dataset_from_directory(..., samples_per_class=1000, random_state=42)`.

Catatan: ekstraktor MobileNetV3-Small memakai bobot ImageNet yang dibekukan, jadi mengekstrak fitur sebelum atau sesudah split tidak menimbulkan leakage parameter. Tetap lakukan split sebelum `StandardScaler.fit`, `PCA.fit`, dan FCM. Sampling tetap harus acak ber-seed.

## 3. Dua jalur ekstraksi fitur

Kedua jalur, setelah ekstraksi, mengikuti alur yang sama: StandardScaler (fit di latih) → PCA (fit di latih) → FCM + KNN atau KNN saja.

### 3.1 Jalur utama (Skenario 1) — MobileNetV3-Small

```python
tf.keras.applications.MobileNetV3Small(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    include_preprocessing=True,
    input_shape=(224, 224, 3),
)
```

- Resize RGB ke 224×224. Kirim tensor **float pada rentang `[0, 255]`**.
- **Jangan** rescale ke `[-1, 1]` atau membagi 255 di luar model. `include_preprocessing=True` sudah merespons Rentang `[0, 255]` di dalam graf Keras. Fungsi `mobilenet_v3.preprocess_input` hanya pass-through jika `include_preprocessing=True`.
- Output: vektor **576** dimensi. Bobot dibekukan (`trainable=False`); model hanya dipakai sebagai ekstraktor, tidak dilatih ulang.
- Cache model (`get_mobilenet_model()` dan/atau `@st.cache_resource`). Usahakan ekstraksi batch di training Colab (jangan prediksi satu citra per panggilan untuk 6.000 gambar, jika mudah diperbaiki).

### 3.2 Jalur pembanding (Skenario 2) — Histogram HSV + GLCM

Pakai OpenCV dan/atau scikit-image (implementasi NumPy untuk HSV boleh dipertahankan asal urutan fitur tetap):

- 16 histogram HSV (8 bin H + 4 bin S + 4 bin V, masing-masing dinormalisasi jumlah = 1)
- 6 statistik warna (mean & std H, S, V)
- 4 tekstur GLCM (contrast, homogeneity, energy, correlation)
- Canny edge density hanya jika `use_edge_density=True`
- Total ~26 dimensi (27 jika tepi aktif)

Jika jalur ini me-resize ke 128×128 (bukan 224×224), dokumentasikan alasan itu di docstring / README: itu pilihan jalur handcrafted, bukan praproses MobileNetV3.

## 4. Pipeline ML full library (wajib sama di kedua jalur)

1. `StandardScaler` (`sklearn.preprocessing`): fit hanya di `X_train`, transform data uji / gambar baru.
2. `PCA` (`sklearn.decomposition`): fit hanya di data latih terskala, **pada kedua jalur**. Default resmi `n_components: 50`. Config juga memuat `variance_threshold: 0.95` sebagai **pemeriksaan**, bukan angka yang boleh diabaikan diam-diam:
   - Tetap pakai 50 komponen sebagai default (usulan dokumen).
   - Simpan `explained_variance_ratio`, `cumulative_variance_ratio`, dan `total_variance_explained` di metrik / `meta.json`.
   - Dukung juga `n_components` berupa float (misalnya `0.95`) dengan batas aman `min(n_components, n_samples-1, n_features)` dan minimum 2 untuk scatter PC1–PC2.
   - Jangan menampilkan “Ambang 90%” di GUI; ambang pemeriksaan adalah 95%.
3. **FCM via `scikit-fuzzy`** (bukan implementasi utama NumPy manual):
   - Latih: `skfuzzy.cluster.cmeans` pada `X_train_pca.T` berbentuk `(n_features, n_samples)`.
   - Keluaran `u` berbentuk `(c, n_samples)` → wajib di-transpose menjadi `(n_samples, c)` sebelum digabung ke KNN.
   - `cntr` (`V`) berbentuk `(c, n_features)` dan disimpan ke `fcm_centers.npy`.
   - Data uji / gambar baru: `skfuzzy.cluster.cmeans_predict(X_new_pca.T, cntr, ...)` dengan pusat **tetap**; jangan fit ulang FCM. Ambil `u.T`.
   - Parameter default: `c=6`, `m=2.0` (simbol `w`), `error=1e-5` (`ξ`), `maxiter=200` (rentang usulan dokumen 100–300), `metric="euclidean"`, `seed=42`.
   - Ambil `jm` (riwayat fungsi objektif `P`), `p` (jumlah iterasi), dan `fpc`.
   - Hitung PE secara eksplisit (basis `ln`): `PE = -(1/n) Σ μ_ik ln(μ_ik)`, suku `μ=0` diperlakukan 0.
   - Silhouette (`sklearn.metrics.silhouette_score`) memakai label crisp `argmax μ` di ruang PCA.
   - **Kejujuran tentang kriteria berhenti:** dokumen kelompok menuliskan `|P_t - P_{t-1}| < ξ` (dengan `P_0=0`, mulai setelah iterasi ke-1). Pustaka `scikit-fuzzy` secara bawaan berhenti pada perubahan matriks `U`. Implementasi tetap memakai perilaku pustaka. Jelaskan perbedaan ini di Info Sistem / laporan, jangan mengaku memakai kriteria `P` jika yang berjalan adalah kriteria `U`.
4. Integrasi fitur: `np.hstack([X_pca, mu * alpha])`. Default `alpha=1.0` (alias `mu_weight`). Nilai `α` dituning di data latih agar `μ ∈ [0, 1]` tidak tertutup skala PCA.
5. `KNeighborsClassifier` (`sklearn`): `metric="euclidean"`, default `k=5`, `weights` `"uniform"` atau `"distance"`. Label berasal dari dataset, bukan dari FCM.

## 5. Tabel parameter resmi

| Parameter | Nilai / rancangan | Keterangan |
| --- | --- | --- |
| Jumlah cluster `c` | 6 | Diselaraskan dengan jumlah kelas; FCM tidak membaca label |
| Bobot fuzziness `w` / `m` | 2.0 | Mempermudah penjelasan rumus `μ` |
| Ambang `ξ` / `error` | `1e-5` | Usulan dokumen; lihat catatan kriteria berhenti di atas |
| `maxiter` | 200 (usulan 100–300) | Batas pengaman |
| Inisialisasi `U` | acak, jumlah tiap sampel = 1, `seed=42` | Agar percobaan dapat diulang |
| `n_components` PCA | 50 (default); periksa ~95% varians | Dipilih setelah melihat kurva varians, lalu dilaporkan apa adanya |
| Bobot integrasi `α` | dituning di data latih | Kandidat usulan: `{0.5, 1.0, 1.5, 2.0}` |
| `k` pada KNN | dicoba `{3, 5, 7, 9}` | Dipilih lewat CV / validasi pada data latih saja |
| Metrik jarak KNN | Euclidean | Di ruang `[PCA || α·μ]` |
| Bobot tetangga | `"uniform"` atau `"distance"` | Dipilih bersama `k` dan `α` |
| Ukuran citra MobileNetV3 | 224×224, piksel `[0, 255]` | `include_preprocessing=True` |
| Rasio latih:uji | 80:20, stratifikasi | 4.800 / 1.200 |
| Seed | 42 | Sampling, split, dan inisialisasi yang perlu diulang |

`config.yaml` di `staging` sudah dekat dengan tabel ini. Jangan mengubah default resmi ke 100 sampel/kelas, `k` hasil tebakan, atau PCA 10 komponen pada training resmi (nilai lebih kecil hanya boleh untuk demo GUI, dan harus dilabeli sebagai demo).

## 6. Rumus internal yang wajib dapat dijelaskan

Simbol `w` pada rumus sama dengan parameter `m` di kode.

**Pusat cluster**

```text
V_kj = [ Σ_i (μ_ik^w · X_ij) ] / [ Σ_i μ_ik^w ]
```

**Jarak kuadrat**

```text
d_ik^2 = Σ_j (X_ij - V_kj)^2
```

**Fungsi objektif**

```text
P = Σ_i Σ_k (μ_ik^w · d_ik^2)
```

**Pembaruan derajat keanggotaan**

```text
μ_ik = (d_ik^2)^(-1/(w-1)) / Σ_j (d_ij^2)^(-1/(w-1))
```

Jika `w = 2`:

```text
μ_ik = (1 / d_ik^2) / Σ_j (1 / d_ij^2)
```

Jumlah `μ` pada setiap baris selalu 1. Lindungi kasus jarak 0 (data tepat di pusat).

**Derajat keanggotaan sampel baru `x`** (pusat `V` tidak dihitung ulang):

```text
d_k^2(x) = Σ_j (x_j - V_kj)^2
μ_k(x)   = (1 / d_k^2(x)) / Σ_j (1 / d_j^2(x))    , untuk w = 2
```

**Jarak Euclidean KNN** di ruang gabungan `[PCA || α·μ]`:

```text
d(X, Y) = √[ Σ_j (X_j - Y_j)^2 ]
```

Kelas diputuskan suara terbanyak dari `k` tetangga (atau berbobot jarak jika `weights="distance"`).

**FPC**

```text
FPC = (1 / n) Σ_i Σ_k μ_ik^2      ∈ [1/c, 1]
```

Makin mendekati 1, partisi makin tegas.

**PE** (basis log natural)

```text
PE = − (1 / n) Σ_i Σ_k μ_ik ln(μ_ik)
```

Jika `μ_ik = 0`, suku itu = 0. Makin kecil PE, makin kecil ketidakpastian.

**Silhouette:** `s(i) = [b(i) − a(i)] / max{a(i), b(i)}`. Label cluster = `argmax μ`. Nilai sistem adalah rata-rata `s(i)`.

**Metrik klasifikasi** (enam kelas): akurasi, presisi, recall, F1 (per kelas serta makro / berbobot), dan confusion matrix.

```text
Akurasi  = (prediksi benar) / (jumlah data uji)
Presisi  = TP / (TP + FP)
Recall   = TP / (TP + FN)
F1       = 2 · (presisi · recall) / (presisi + recall)
```

## 7. Contoh perhitungan manual resmi (Kilap & Coklat)

Contoh ini wajib sama di `notebooks/training.ipynb`, README, dan (ringkas) Info Sistem. Jangan mengganti matriks `U` awal.

Lima data latih, dua fitur analog PCA (“Kilap”, “Coklat”), skala 0–10:

| Gambar | Kilap | Coklat | Label |
| --- | --- | --- | --- |
| A | 7 | 1 | plastik |
| E | 6 | 2 | plastik |
| B | 9 | 1 | kaca |
| C | 1 | 9 | kardus |
| D | 2 | 7 | kertas |

Parameter: `c=2`, `w=2`, `ξ=0.001`, `P_0=0`, `MaxIter=100`, `k=3`. FCM tidak memakai kolom label.

Matriks `U` awal (jumlah tiap baris = 1):

| Gambar | μ₁ | μ₂ |
| --- | --- | --- |
| A | 0,6 | 0,4 |
| E | 0,3 | 0,7 |
| B | 0,5 | 0,5 |
| C | 0,4 | 0,6 |
| D | 0,7 | 0,3 |

Gambar baru resmi: **X = (Kilap=6, Coklat=3)**. Hitung `μ` dari `V` akhir (tanpa iterasi FCM ulang), gabungkan menjadi `(6; 3; μ₁; μ₂)`, lalu KNN `k=3`. Menurut dokumen kelompok: `μ₁≈0,092`, `μ₂≈0,908`, tiga tetangga terdekat E, A, B, prediksi **plastik** (2 suara vs 1), dengan interpretasi 90,8% ke kelompok mengkilap dan 9,2% ke kelompok coklat.

Nomor cluster dapat tertukar tergantung `U` awal; yang penting pola keanggotaannya. Label selalu dari dataset, bukan dari FCM.

## 8. Kontrak kode `src/` (yang sudah ada dan yang harus dirapikan)

API yang sudah ada dan wajib dipertahankan:

- `extract_features`, `get_mobilenet_model`, `extract_handcrafted_features`
- `FCM`, `fcm_predict_membership`, `compute_fpc`, `compute_partition_entropy` / `compute_pe`, `compute_fcm_silhouette`
- `load_dataset_from_directory`, `load_default_config`
- `fit_pipeline(X, y=None, config=None, feature_extractor=None)`
- `transform`, `predict`, `evaluate_pipeline`
- `tune_knn_hyperparameters` (CV di data latih saja)
- `run_four_scenarios` (split yang sama untuk 1a/1b/2a/2b)
- `save_models`, `load_models`

Perbaikan yang masih diperlukan di tahap ini (detail skenario dan `meta.json` ada di rule 02, detail GUI di rule 03):

- `save_models` di `staging` men-reset `meta_dict` di tengah fungsi, sehingga `class_names`, `feature_extractor`, `alpha`, dan `scenarios_summary` bisa hilang. Perbaiki agar seluruh metadata resmi tersimpan.
- `fit_pipeline` membaca `variance_threshold` tetapi tidak memakainya. Laporkan varians 50 komponen, dan dukung float `n_components`.
- Jangan menelan kegagalan `run_four_scenarios` dengan `except Exception: pass` tanpa jejak.
- Notebook dan `app.py` masih memanggil tanda tangan / nama fungsi yang tidak ada (`y_true=`, `test_size=`, `compare_scenarios`, `compare_knn_variants`, `compare_feature_extractors`). Samakan ke API di atas. Alias tipis boleh ditambahkan hanya jika tidak menduplikasi logika.
- Usahakan ekstraksi MobileNetV3 secara batch pada training resmi.

## 9. Konsistensi versi pustaka

Samakan versi antara Colab, laptop, `requirements.txt`, Info Sistem, dan `meta.json["library_versions"]`:

- `scikit-learn==1.6.1`
- `scikit-fuzzy==0.5.0`
- `tensorflow>=2.15.0`
- `numpy>=1.24.0,<2.0.0`
- `joblib>=1.3.2`

Tambahkan `pandas` jika tabel skenario memakainya. `seaborn` / `plotly` boleh ditambahkan **hanya jika benar-benar dipakai**; jika visualisasi tetap Matplotlib, jangan memaksa dependensi mati. Jangan menyisakan `scikit-learn==1.4.2`, MobileNetV2, 1280 fitur, TrashNet, atau placeholder `<username>`.

Keputusan official (untuk diselaraskan di laporan jika teks lama masih menyebut “FCM ditulis manual dengan NumPy”): dosen mengizinkan pustaka siap pakai, asal proses algoritma dipahami, dijelaskan, dan didokumentasikan. Kode inti yang ditunjukkan saat demo adalah wrapper `src/fcm.py` beserta rumus di atas.
