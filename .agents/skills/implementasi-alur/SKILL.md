---
name: implementasi-alur
description: Menyelesaikan implementasi alur TLF (FCM + KNN) secara cepat di atas kode staging yang sudah ada. Pakai saat merapikan src/, menyambungkan app.py dan notebook Colab, menjalankan 5 skenario, atau memperbaiki kontrak meta.json — bukan untuk setup ulang, training Kaggle penuh di lokal, atau verifikasi panjang.
---

# Implementasi alur (cepat, dari staging)

Skill ini untuk **menyelesaikan alur yang sudah ter-scaffold**. Basisnya adalah branch `staging` (commit `367fb93`). Jangan mengulang setup, jangan mendirikan folder baru dari nol, dan jangan mengunduh dataset Kaggle penuh di mesin lokal.

Ikuti `@AGENTS.md` plus tiga rule di `.agents/rules/`. Ringkas temuan audit `staging` ada di `PROMPT-IMPLEMENTASI.md`.

## 1. Rencana singkat, lalu 3 worker paralel

Tulis rencana beberapa bullet (bukan dokumen panjang). Pecah ke worker yang **tidak mengedit file yang sama**:

| Worker | File yang boleh disentuh | Jangan sentuh |
| --- | --- | --- |
| A — inti & skenario | `src/__init__.py`, `src/features.py`, `src/fcm.py`, `src/pipeline.py`, `config.yaml` | `app.py`, `notebooks/`, README |
| B — GUI | `app.py` | `src/`, `config.yaml` (boleh baca) |
| C — runner & docs | `notebooks/training.ipynb`, `requirements.txt`, `README.md`, `data/README.md`, `models/README.md` | logika baru di `src/`, halaman di `app.py` |

Tetapkan dulu API yang sudah ada agar B dan C tidak “menebak” tanda tangan:

- `fit_pipeline(X, y=None, config=None, feature_extractor=None)`
- `run_four_scenarios(...)`, `tune_knn_hyperparameters(...)`
- `evaluate_pipeline(...)` mengembalikan `classification_metrics`
- `save_models` / `load_models` dengan `meta.json` utuh (lihat rule 02)
- Kunci hasil: `class_names`, `data.X_train_pca`, `metrics.fcm_train`, `metrics.knn_test`, `metrics.scenarios`

Jangan menyalin atau menimpa `AGENTS.md`, `CARA-PAKAI.md`, `PROMPT-IMPLEMENTASI.md`, atau isi `.agents/` milik paket ini.

## 2. Tugas konkret per worker (berdasarkan audit `staging`)

### Worker A — `src/` & `config.yaml`

- Pertahankan full library: MobileNetV3-Small `[0, 255]` 576-d, HSV+GLCM, `skfuzzy.cmeans` / `cmeans_predict` dengan transpose yang benar, `[PCA || alpha * mu]`, KNN Euclidean.
- Bekukan bobot MobileNetV3-Small. Jika mudah, sediakan ekstraksi batch untuk training Colab.
- Perbaiki `save_models`: jangan reset `meta_dict`; persist `class_names`, `feature_extractor`, `alpha`, metrik, skenario, tuning, versi pustaka sesuai kontrak rule 02.
- `fit_pipeline` / `run_four_scenarios`: laporkan varians PCA 50 komponen vs ambang 0,95; dukung `n_components` float; jangan `except: pass` yang menelan kegagalan 4 skenario.
- Pastikan `tune_knn_hyperparameters` siap dipanggil (CV latih saja) dan, jika perlu, tambahkan helper Skenario 4 (opsional, `c`/`m`) serta ringkasan Skenario 5 (per kelas + matrix) tanpa menduplikasi logika evaluasi.
- Jangan mengubah default resmi di `config.yaml` (1000/kelas, split 0.2, `c=6`, `m=2`, `maxiter=200`, `error=1e-5`, `k=5`, `alpha=1.0`, PCA 50).
- Alias tipis (`compare_scenarios` sebagai pembungkus `run_four_scenarios`) hanya boleh jika tidak menduplikasi pipeline. Prioritas: API yang sudah ada tetap benar, lalu notebook/app disesuaikan.

### Worker B — `app.py`

- Perbaiki impor agar `SRC_AVAILABLE` tidak jatuh hanya karena nama fungsi usang.
- Perbaiki pemanggilan `fit_pipeline(y=...)` dan pemetaan kunci metrik (lihat rule 02).
- **Hapus semua angka fiktif** (akurasi cadangan, confusion matrix hardcode, tabel 94,80% / 89,20% / 81,50% / 75,30%). Jika data belum ada, tampilkan keadaan kosong yang jujur.
- Enam halaman lengkap sesuai rule 03, termasuk validasi `c < 2`, `m <= 1`, `k < 1`, berkas kosong/rusak, format JPG/JPEG/PNG.
- Visualisasi: tabel `μ`, scatter PC1–PC2 + pusat, kurva `P`, confusion matrix nyata, tabel 4 skenario dari `meta.json` / `metrics["scenarios"]`.
- Sambungkan ZIP / dataset unggahan ke pelatihan demo. Labeli data sintetis sebagai demo.
- Hindari split ganda. Latih ulang hanya di `st.session_state`, jangan menimpa `models/`.
- Info Sistem: full library + rumus + FAQ dosen + versi `scikit-learn` / `scikit-fuzzy` + contoh Kilap/Coklat resmi + ambang varians 95% (bukan 90%).
- Cache MobileNetV3-Small; usahakan satu sumber model dengan `src.features`.

### Worker C — notebook, requirements, README

- Ubah `notebooks/training.ipynb` menjadi runner Colab tipis yang **mengimpor `src/`**, siap dijalankan tanpa sel yang langsung error.
- Aktifkan langkah setup Colab: clone repo yang sedang dikerjakan, `%cd` ke root, `pip install -r requirements.txt`.
- Cek versi pustaka vs `requirements.txt` (`scikit-learn==1.6.1`, `scikit-fuzzy==0.5.0`).
- Pakai `load_dataset_from_directory(..., samples_per_class=1000, random_state=42)`, bukan `img_paths[:N]` dan bukan default 100/30 untuk eksperimen resmi.
- Contoh manual Kilap/Coklat: `U` awal resmi, `ξ=0.001`, `k=3`, prediksi X=(6, 3). Jangan memakai matriks `U` 0,85/0,15 dst.
- Panggil `fit_pipeline(X, y=y, config=config)`; baca `results["data"]`, `results["metrics"]`, `results["class_names"]`.
- Jalankan Skenario 3 (`tune_knn_hyperparameters` di latih), lalu 1a/1b/2a/2b pada split yang sama (`run_four_scenarios`), Skenario 4 opsional, Skenario 5 dari `per_class` + confusion matrix. Jangan mengimpor `compare_scenarios` kecuali alias resmi sudah ada.
- Simpan artefak lewat `save_models` dengan `meta=` utuh, lalu verifikasi `load_models`.
- Tambahkan `pandas` (dan seaborn/plotly hanya jika dipakai) ke `requirements.txt`. Rapikan URL clone di README agar konsisten dengan repo kerja, tanpa menulis ulang README dari nol.

## 3. Smoke check ringan, lalu STOP

Hanya ini, dan hanya sekali, **tanpa** mengunduh Kaggle dan **tanpa** membuka browser:

1. `python -m py_compile app.py src/features.py src/fcm.py src/pipeline.py`
2. Impor publik berhasil: `from src.pipeline import fit_pipeline, run_four_scenarios, tune_knn_hyperparameters, save_models, load_models, evaluate_pipeline` dan `from src.fcm import FCM, fcm_predict_membership`.
3. Uji sintetis kecil (misalnya 12 citra acak / 2 per kelas, atau matriks fitur palsu 26-d jika TensorFlow/bobot ImageNet tidak tersedia): `StandardScaler → PCA → skfuzzy FCM → [PCA || alpha*mu] → KNN`, plus `cmeans_predict` untuk data baru, plus `run_four_scenarios` dan `tune_knn_hyperparameters` pada data latih, plus `save_models` / `load_models` ke folder sementara dan cek `meta.json` memuat `class_names`, `library_versions`, dan kerangka `scenarios`.
4. Pencarian teks: tidak ada sisa `MobileNetV2`, `1280`, `TrashNet`, `1.4.2`, atau `<username>`. Tidak ada akurasi hardcode `94.80`, `0.9480`, `89.20`, `81.50`, `75.30`, `0.9167` di `app.py`.
5. `app.py` tidak mengimpor nama yang tidak ada di `src.pipeline`.

**Jangan** menambah pytest suite, **jangan** training 6.000 citra di lokal, **jangan** audit berulang, **jangan** merapikan gaya di luar yang rusak.

Setelah smoke check lulus atau setelah satu perbaikan cepat jika ada error jelas: berhenti. Laporkan file yang diubah, bug yang diperbaiki, perintah menjalankan notebook di Colab, dan hasil smoke check.
