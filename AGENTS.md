# Aturan workspace — Tahap IMPLEMENTASI Alur

Final Project TLF Kelompok 5 (Kelas F) — Klasifikasi Sampah Berbasis FCM + KNN.

Dosen pengampu: Dr. Made Agung Raharja, S.Si., M.Cs.  
Program Studi Informatika, FMIPA, Universitas Udayana.

**Tahap setup sudah selesai.** Basis kerja adalah kode di branch `staging` repo [Leoudah/Garbage-Classification-FCM](https://github.com/Leoudah/Garbage-Classification-FCM) (commit `367fb93`, 24 Nov 2026, 12:25 WITA), hasil merge `dev/deas`. Jangan merancang ulang proyek dari nol dan jangan mengulang scaffolding.

Tugas sekarang: **mengimplementasikan, merapikan, menyambungkan, dan memverifikasi alur lengkap** agar repo, GUI, notebook, dan artefak `models/meta.json` sinkron dengan Dokumen Konteks Sistem serta Contoh Perhitungan Manual. Pustaka boleh dipakai penuh (`scikit-fuzzy`, TensorFlow/Keras, scikit-learn); rumus internal tetap wajib dipahami dan dijelaskan.

Acuan aturan:

- Arsitektur, parameter, dan rumus: `.agents/rules/01-konteks-arsitektur-pipeline.md`
- Skenario 1–5, kontrak `meta.json`, larangan angka fiktif: `.agents/rules/02-skenario-pengujian-evaluasi.md`
- Enam halaman Streamlit dan keamanan demo: `.agents/rules/03-gui-streamlit-demo.md`
- Alur kerja cepat: skill `implementasi-alur` di `.agents/skills/implementasi-alur/SKILL.md`

## Mode kerja: `/swarm` cepat

- Pakai **`/swarm`**. Buat rencana singkat (beberapa bullet), bagi kerja ke 3 worker paralel, lakukan smoke check ringan, lalu **STOP**.
- **Jangan pakai `/teamwork` default.** Mode itu membawa interview, milestone panjang, serta gerbang Critic / Challenger / Auditor. Tidak diperlukan pada tahap ini.
- Kalau sesi sudah masuk `/teamwork`: pakai integrity `development`, lewati interview, jangan pindah ke `~/teamwork_projects/`, kerjakan di workspace ini, jangan spawn Critic/Challenger/Auditor, cukup smoke check, lalu berhenti.
- Jangan menulis test suite pytest yang lengkap, jangan verifikasi UI lewat browser, dan **jangan mengunduh dataset Kaggle penuh** saat pengembangan lokal.
- Jangan menimpa `AGENTS.md`, `CARA-PAKAI.md`, `PROMPT-IMPLEMENTASI.md`, atau isi `.agents/` milik paket ini, kecuali ada kesalahan nyata di berkas itu.

Pembagian kerja yang disarankan (file ownership jangan tumpang tindih):

1. Worker A — `src/` + `config.yaml` (inti pipeline, skenario, tuning, kontrak `meta.json`)
2. Worker B — `app.py` (6 halaman GUI, validasi, visualisasi, pemetaan metrik, hapus angka fiktif)
3. Worker C — `notebooks/training.ipynb`, `requirements.txt`, `README.md`, `data/README.md`, `models/README.md` (runner Colab tipis dan konsistensi versi)

## Peta repo dan batas wewenang

Logika resmi hanya hidup di `src/`, `app.py`, dan `config.yaml`. Notebook adalah **runner eksperimen tipis** yang mengimpor `src/`, bukan tempat menyalin ulang algoritma.

| Lokasi | Peran |
| --- | --- |
| `src/features.py` | MobileNetV3-Small (576-d, piksel float `[0, 255]`, `include_preprocessing=True`) dan jalur HSV + GLCM |
| `src/fcm.py` | Wrapper `skfuzzy.cluster.cmeans` & `cmeans_predict` (plus fallback NumPy), metrik FPC / PE / Silhouette |
| `src/pipeline.py` | Sampling, split anti-leakage, scaler, PCA, integrasi `[PCA || alpha * mu]`, KNN, 4 skenario, tuning, `save_models` / `load_models` |
| `config.yaml` | Parameter default resmi |
| `app.py` | GUI Streamlit 6 menu; impor API dari `src/`; latih ulang demo hanya di `st.session_state` |
| `notebooks/training.ipynb` | Runner Colab: unduh data, contoh manual Kilap/Coklat, training resmi, skenario 1–5, simpan `models/` |
| `models/` | Artefak resmi hasil Colab (`scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, `meta.json`). Folder ini masih kosong sampai training dijalankan. |

## Guardrail teknis (jangan dilanggar)

1. **Anti data leakage.** Sampling acak 1.000 citra per kelas (`seed=42`, total 6.000), lalu split stratified 80:20 (`seed=42` → 4.800 latih / 1.200 uji) **paling awal**, sebelum `StandardScaler.fit`, `PCA.fit`, dan FCM. Data uji 1.200 **tidak boleh** dipakai untuk memilih `k`, `weights`, `alpha`, `c`, atau `m`.
2. **Full library, rumus tetap dijelaskan.** FCM resmi memakai `scikit-fuzzy` (`cmeans`, `cmeans_predict`). Hormati kontrak transpose `(n_features, n_samples)` ⟷ `(n_samples, c)`. Jangan mengembalikan implementasi utama ke FCM manual NumPy. Fallback NumPy boleh tetap ada, tetapi jalur utama adalah pustaka. Info Sistem, README, dan notebook wajib menjelaskan rumus `V`, `d²`, `P`, `μ`, FPC, PE, dan Silhouette.
3. **Jangan menimpa `models/` dari GUI.** Latih ulang di halaman Proses hanya disimpan di `st.session_state`. Artefak disk hanya ditulis oleh notebook Colab / `save_models()` sadar.
4. **Jangan membuat angka hasil.** Dilarang mengisi akurasi, F1, FPC, confusion matrix, atau tabel 4 skenario dengan nilai rekaan. Jika eksperimen belum dijalankan, tampilkan keadaan kosong yang jujur (misalnya “belum ada hasil eksperimen”).
5. **Samakan tanda tangan API.** Notebook dan `app.py` wajib memanggil fungsi `src/` sesuai parameter yang benar (`y=`, bukan `y_true=` pada `fit_pipeline`; `run_four_scenarios`, bukan fungsi yang tidak ada). Impor yang gagal tidak boleh membuat seluruh `src` “menghilang” secara diam-diam.
6. **Versi pustaka harus selaras** antara `requirements.txt`, Colab, lokal, dan `meta.json` (`scikit-learn==1.6.1`, `scikit-fuzzy==0.5.0`, `tensorflow>=2.15.0`). Jangan menyisakan `scikit-learn==1.4.2`.
7. **Teks UI berbahasa Indonesia.** Identifier kode, nama file, dan kunci `config.yaml` tetap bahasa Inggris.
8. **Contoh manual resmi** di notebook, README, dan Info Sistem wajib memakai data Kilap/Coklat dari dokumen kelompok (termasuk matriks `U` awal yang sama), bukan angka inisialisasi lain.

Teks UI, komentar, dan dokumentasi merujuk arsitektur: **MobileNetV3-Small + StandardScaler + PCA + scikit-fuzzy FCM + KNN `[PCA || α·μ]`**. Jangan mengembalikan istilah lama MobileNetV2, 1280 fitur, TrashNet, atau `scikit-learn==1.4.2`.
