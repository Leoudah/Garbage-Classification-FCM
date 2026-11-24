---
name: setup-scaffold
description: Scaffolds the TLF Final Project Streamlit repo (PCA + FCM + KNN) in fast parallel mode. Use when setting up the project, creating src/, app.py, the Colab notebook, or README — not for full training, report writing, or heavy verification.
---

# Setup scaffold (cepat)

Skill ini hanya untuk **mendirikan kerangka repo**. Bukan untuk training dataset penuh, bukan untuk laporan BAB I–V, dan bukan untuk loop verifikasi panjang.

## 1. Rencana singkat, lalu paralel

Tulis rencana beberapa bullet (bukan dokumen panjang). Langsung pecah ke worker / anggota swarm yang **tidak mengedit file yang sama**:

| Worker | File yang boleh disentuh | Jangan sentuh |
| --- | --- | --- |
| A — algoritma | `src/__init__.py`, `src/features.py`, `src/fcm.py`, `src/pipeline.py`, `config.yaml` | `app.py`, `notebooks/`, `README.md` |
| B — GUI | `app.py`, `pages/` (hanya jika perlu) | `src/`, `config.yaml` (boleh baca) |
| C — notebook & docs | `notebooks/training.ipynb`, `README.md`, `requirements.txt`, `.gitignore`, `data/`, `models/` (placeholder saja) | logika di `src/`, halaman di `app.py` |

Worker B dan C **import / merujuk** API yang disepakati dari `src/pipeline.py`, `src/features.py`, dan `src/fcm.py`. Nama fungsi utama: `extract_features`, `fit_pipeline`, `transform`, `predict`, `load_models`, `save_models`, `fcm_predict_membership`, `compare_knn_variants`, `compare_feature_extractors`.

## 2. Yang harus ada di kode

Ikuti @SPESIFIKASI-PROYEK.md dan rule `.agents/rules/algoritma-pipeline.md`. Ringkasnya:

- Dataset acuan: Kaggle "Garbage Images Dataset" (`zlatan599/garbage-dataset-classification`).
- Ekstraksi fitur: MobileNetV2 1280D utama, Handcrafted HSV+GLCM ~26D pembanding.
- Anti data-leakage: Stratified split sebelum fitting StandardScaler, PCA, dan FCM.
- FCM manual NumPy di `src/fcm.py` ($U$, centers, objective history, FPC, PE, `fcm_predict_membership`).
- KNN terpadu dengan input `[fitur_pca, mu * mu_weight]` dilatih dengan label kelas aktual.
- Streamlit: 6 halaman, validasi upload, latih ulang di `st.session_state` (jangan overwrite `models/`), prediksi foto baru, evaluasi dan 2 tabel perbandingan.
- Notebook Colab: import dari `src/`, stratified training, contoh hitungan manual FCM (lengkap dengan FPC & PE), 2 sel perbandingan, sel simpan `models/` + `meta.json`.
- README: cara install, `streamlit run app.py`, alur Colab, catatan versi scikit-learn & tensorflow.

## 3. Smoke check, lalu STOP

1. Import `src.features`, `src.fcm`, `src.pipeline` berhasil.
2. Jalankan pipeline pada data sintetis kecil (FCM dan KNN tidak boleh crash).
3. `streamlit run app.py` berhasil start tanpa error import.
4. Jika model `.pkl` belum ada di `models/`, UI harus menampilkan pesan ramah, bukan traceback.
