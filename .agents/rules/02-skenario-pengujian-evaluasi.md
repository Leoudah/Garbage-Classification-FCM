---
trigger: glob
globs: "src/**/*.py, notebooks/**/*.ipynb, app.py, models/**, config.yaml"
description: "Aturan Skenario 1-5, kontrak models/meta.json, dan larangan merakit angka hasil eksperimen."
---

# Skenario pengujian, evaluasi, dan kontrak `meta.json`

Semua skenario merujuk pada **pembagian data yang sama**: sampling 1.000/kelas `seed=42`, split stratified 80:20 `seed=42` (4.800 latih / 1.200 uji). Setelah ekstraksi, alurnya sama: StandardScaler (fit di latih) → PCA (fit di latih) → FCM + KNN atau KNN saja.

Angka hasil **tidak boleh dikira-kira**. Tabel di Dokumen Konteks Sistem masih bertuliskan “(diisi setelah eksperimen)”. Isi resmi baru sah setelah `notebooks/training.ipynb` dijalankan di Colab pada dataset penuh, lalu disimpan ke `models/meta.json`.

## Larangan angka fiktif (wajib)

Dilarang mengisi UI, notebook, README, atau `meta.json` dengan akurasi / F1 / FPC / confusion matrix rekaan.

Temuan di `staging` yang **harus dihapus atau diganti keadaan kosong yang jujur**:

- Fallback `app.py` pada halaman Evaluasi dan pesan sukses Proses, misalnya default `0.9480`, `0.9495`, `0.9485`, `0.9167`, serta tabel perbandingan 1a/1b/2a/2b dengan akurasi `94.80%`, `89.20%`, `81.50%`, `75.30%`.
- Confusion matrix hardcode 6×6 ketika data nyata belum ada.
- Narasi “kesalahan terbanyak terjadi pada `trash`” yang ditulis seolah hasil sudah diukur, padahal belum ada eksperimen resmi.

Jika artefak atau metrik belum tersedia, tampilkan pesan seperti: **“Hasil eksperimen belum tersedia. Jalankan `notebooks/training.ipynb` di Google Colab, lalu muat folder `models/`.”** Jangan menampilkan persentase palsu “agar halaman tidak kosong”.

## Skenario 1 — Pengaruh integrasi FCM pada fitur deep

| Kode | Konfigurasi | Tujuan |
| --- | --- | --- |
| **1a** | MobileNetV3-Small + PCA + (FCM + KNN) | Model usulan: masukan KNN = `[X_PCA || α·μ]` |
| **1b** | MobileNetV3-Small + PCA + KNN | Baseline ablasi: KNN hanya pada `X_PCA`, tanpa `μ` |

Kriteria diterima:

- Kedua konfigurasi memakai split, scaler, dan PCA yang sama pada jalur MobileNetV3-Small.
- 1a dan 1b dilaporkan bersebelahan: akurasi, presisi makro, recall makro, F1 makro (plus berbobot jika ada), dan confusion matrix.
- 1a juga menyimpan metrik FCM di data latih: `P` akhir, jumlah iterasi, FPC, PE, Silhouette.
- Jika 1a tidak lebih baik dari 1b, hasil tetap dilaporkan jujur beserta analisis penyebab (misalnya `α` terlalu kecil, `μ` kurang informatif, atau PCA sudah sangat memisahkan kelas).

## Skenario 2 — Fitur handcrafted vs deep, dengan dan tanpa FCM

| Kode | Konfigurasi | Tujuan |
| --- | --- | --- |
| **2a** | (Histogram HSV + GLCM) + PCA + (FCM + KNN) | Integrasi FCM pada fitur warna/tekstur |
| **2b** | (Histogram HSV + GLCM) + PCA + KNN | Baseline handcrafted tanpa `μ` |

Kriteria diterima:

- 2a/2b memakai **split gambar yang sama** dengan 1a/1b (bukan sampling ulang).
- Scaler dan PCA di-fit terpisah pada jalur handcrafted (karena dimensinya berbeda), tetapi protokolnya identik.
- Tabel empat konfigurasi 1a, 1b, 2a, 2b wajib utuh di notebook, `meta.json`, dan halaman Evaluasi.

Fungsi `run_four_scenarios(...)` di `staging` sudah mengarah ke sini. Rapikan:

- Jangan mengekstrak ulang MobileNetV3 secara sia-sia jika fitur jalur 1 sudah dihitung di `fit_pipeline`; boleh menerima matriks fitur yang sudah diekstrak, atau men-cache hasil ekstraksi.
- Jangan menelan error dengan `except Exception: pass` tanpa mencatat kegagalan.
- Simpan juga `P` akhir dan `n_iter` pada 1a/2a, bukan hanya FPC/PE/Silhouette.
- Jangan memakai subset 30 atau 100 gambar untuk tabel resmi.

## Skenario 3 — Tuning `k`, `weights`, dan `α` (hanya di data latih)

Tujuan: memilih `k ∈ {3, 5, 7, 9}`, `weights ∈ {"uniform", "distance"}`, dan `α` (kandidat usulan `{0.5, 1.0, 1.5, 2.0}`) plus pemeriksaan varians PCA.

Kriteria diterima:

- **Hanya** validasi silang / fold di **4.800 data latih**. Test set 1.200 **tidak boleh** masuk pencarian parameter.
- Fungsi `tune_knn_hyperparameters(...)` sudah ada di `staging` tetapi **belum dipanggil** dari notebook, `fit_pipeline`, atau GUI. Wajib dijalankan di runner Colab, lalu parameter terbaik dipakai untuk model resmi yang disimpan ke `models/`.
- Simpan seluruh kisi hasil (`cv_results`) dan `best_params` ke `meta.json["tuning"]`.
- Setelah parameter dipilih, evaluasi akhir 1a/1b/2a/2b diukur sekali di data uji (bukan untuk memilih lagi).
- Periksa dan laporkan varians kumulatif PCA 50 komponen terhadap ambang 0,95. Jika 50 komponen belum mencapai 95%, tulis apa adanya; jangan diam-diam mengganti protokol tanpa catatan.

Catatan metodologis (boleh dipertahankan asal didokumentasikan): CV pada `k`/`weights`/`α` boleh memakai `μ` dari FCM yang di-fit pada seluruh data latih. Jangan meng-tune di test set.

## Skenario 4 (opsional) — Kepekaan parameter FCM

Variasikan `c` (misalnya `{4, 6, 8}`) dan/atau `m` (misalnya `{1.5, 2.0, 2.5}`) pada data latih, lalu laporkan `P`, iterasi, FPC, PE, Silhouette, dan jika relevan F1/akurasi KNN hybrid di fold latih.

Kriteria diterima:

- Bersifat opsional, tetapi jika dikerjakan harus memakai protokol yang sama dan **bukan** test set untuk memilih `c`/`m`.
- Default resmi yang disimpan ke `models/` tetap `c=6`, `m=2.0`, kecuali ada alasan empiris yang ditulis jujur.
- Jika dilewati, `meta.json["scenario_4"]` boleh `{"status": "tidak dijalankan"}` — jangan mengisi angka palsu.

## Skenario 5 — Evaluasi per kelas dan analisis `trash`

Tujuan: melihat kelas mana yang paling sering tertukar, khususnya `trash` yang isinya heterogen.

Kriteria diterima:

- Tampilkan presisi, recall, F1 per kelas dan confusion matrix untuk konfigurasi utama (minimal 1a), idealnya juga 1b/2a/2b.
- Analisis `trash` **hanya** berdasarkan matrix / metrik nyata. Jika `trash` memang paling sering keliru, bahas sebagai keterbatasan data dan model, jangan disembunyikan.
- Jika kelas lain justru lebih bermasalah, tulis itu. Jangan mengunci narasi sebelum eksperimen jalan.
- `compute_classification_metrics` di `staging` sudah menghasilkan `per_class` dan `confusion_matrix`. Sambungkan ke notebook, `meta.json`, dan halaman Evaluasi (kunci yang benar, bukan `metrics["accuracy"]` di akar yang tidak ada).

## Kontrak keluaran `models/` dan `meta.json`

Lima artefak resmi:

| File | Isi |
| --- | --- |
| `scaler.pkl` | `StandardScaler` fit di data latih jalur model yang disimpan |
| `pca.pkl` | `PCA` fit di data latih terskala |
| `fcm_centers.npy` | Pusat `V`, shape `(c, n_components)` |
| `knn.pkl` | `KNeighborsClassifier` pada `[X_PCA || α·μ]` |
| `meta.json` | Metadata, versi pustaka, metrik, dan tabel skenario |

`save_models` harus menulis `meta.json` utuh (perbaiki bug reset `meta_dict` dan `**kwargs` yang diabaikan). Struktur minimum:

```json
{
  "saved_at": "ISO-8601",
  "class_names": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "feature_extractor": "mobilenetv3_small",
  "alpha": 1.0,
  "mu_weight": 1.0,
  "n_components": 50,
  "explained_variance_ratio": [],
  "cumulative_variance_ratio": [],
  "total_variance_explained": 0.0,
  "variance_threshold": 0.95,
  "c": 6,
  "m": 2.0,
  "error": 1e-5,
  "maxiter": 200,
  "k": 5,
  "weights": "uniform",
  "metric": "euclidean",
  "random_state": 42,
  "dataset": {
    "slug": "zlatan599/garbage-dataset-classification",
    "samples_per_class": 1000,
    "n_train": 4800,
    "n_test": 1200,
    "test_size": 0.2,
    "random_state": 42
  },
  "library_versions": {
    "numpy": "...",
    "scikit-learn": "1.6.1",
    "scikit-fuzzy": "0.5.0",
    "tensorflow": "...",
    "joblib": "..."
  },
  "fcm_train": {
    "final_objective": 0.0,
    "n_iter": 0,
    "fpc": 0.0,
    "pe": 0.0,
    "silhouette": 0.0
  },
  "knn_test": {
    "accuracy": 0.0,
    "precision_macro": 0.0,
    "recall_macro": 0.0,
    "f1_macro": 0.0,
    "precision_weighted": 0.0,
    "recall_weighted": 0.0,
    "f1_weighted": 0.0,
    "per_class": {},
    "confusion_matrix": [],
    "labels": []
  },
  "scenarios": {
    "1a": {"name": "Skenario 1a: MobileNetV3-Small + PCA + (FCM + KNN)", "metrics": {}, "fcm_metrics": {}, "pca_components": 50},
    "1b": {"name": "Skenario 1b: MobileNetV3-Small + PCA + KNN", "metrics": {}, "pca_components": 50},
    "2a": {"name": "Skenario 2a: (HSV + GLCM) + PCA + (FCM + KNN)", "metrics": {}, "fcm_metrics": {}, "pca_components": 50},
    "2b": {"name": "Skenario 2b: (HSV + GLCM) + PCA + KNN", "metrics": {}, "pca_components": 50}
  },
  "tuning": {
    "best_params": {},
    "cv_results": [],
    "note": "Validasi silang hanya pada data latih (Skenario 3)."
  },
  "scenario_4": {"status": "tidak dijalankan"},
  "scenario_5": {
    "per_class": {},
    "confusion_matrix": [],
    "notes": ""
  },
  "config": {}
}
```

Angka `0.0` di contoh di atas hanyalah placeholder skema, **bukan** nilai yang boleh ditampilkan sebagai hasil. Objek nyata diisi dari keluaran pipeline. Kunci metrik klasifikasi yang sudah dipakai `compute_classification_metrics` wajib dihormati: `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `per_class`, `confusion_matrix` — bukan `precision` / `f1` di akar objek jika kunci itu tidak ada.

## Pemetaan kunci yang salah di `staging` (wajib diperbaiki)

`fit_pipeline` mengembalikan:

- `results["class_names"]` (bukan `"classes"`)
- `results["data"]["X_train_pca"]`, `X_test_pca`, `y_test`, `y_pred_test`, `mu_train`, `mu_test`
- `results["metrics"]["fcm_train"]` → `fpc`, `pe`, `silhouette`, `n_iter`, `final_objective`, `objective_history`
- `results["metrics"]["pca"]` → `explained_variance_ratio`, `cumulative_variance_ratio`, `total_variance_explained`
- `results["metrics"]["knn_test"]` → metrik klasifikasi di atas
- `results["metrics"]["scenarios"]` → `"1a"..."2b"`
- `results["fcm"].labels_`, `results["fcm"].centers_`, `results["fcm"].objective_history_`

`evaluate_pipeline` mengembalikan `{"fpc", "pe", "classification_metrics", "n_samples"}` — bukan `{"accuracy", "precision", "f1"}` di akar.

Notebook `staging` masih memanggil `fit_pipeline(..., y_true=..., test_size=..., random_state=...)` (parameter itu tidak ada), mengimpor `compare_scenarios` (fungsi tidak ada), dan membaca `results["X_pca_train"]`, `results.get("fpc")`, `results["cluster_labels_train"]`. Itu akan gagal saat dijalankan dan harus disambungkan ke kunci di atas.

Halaman Evaluasi / Proses di `app.py` juga membaca `metrics.get("fpc")` dan `test_eval.get("accuracy")`, lalu jatuh ke default fiktif. Samakan ke struktur ini, dan jika kunci absen tampilkan keadaan kosong — bukan angka cadangan.
