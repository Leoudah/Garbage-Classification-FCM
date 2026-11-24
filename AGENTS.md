# Aturan proyek — Final Project TLF (Fuzzy Sampah)

Aplikasi Python + Streamlit untuk **pengelompokan & klasifikasi foto sampah yang belum berlabel**, dengan alur **MobileNetV2 + PCA + Fuzzy C-Means (FCM) + KNN (+ mu)**.

Spesifikasi lengkap: @SPESIFIKASI-PROYEK.md  
Saat scaffolding/pengembangan repo, ikuti skill `setup-scaffold` di `.agents/skills/setup-scaffold/SKILL.md`.

## Mode kerja: teamwork cepat

Tugas di repo ini adalah pengembangan modul terpadu secara efisien:

- Pakai **`/swarm`** atau subagent paralel biasa. Pecah kerja ke beberapa worker, file ownership jangan tumpang tindih, lalu smoke check, lalu berhenti.
- **Jangan pakai `/teamwork` default.** Mode itu membawa interview Phase 1, rencana milestone panjang, dan gerbang verifikasi (Critic, Challenger, Auditor, Success Auditor). Tidak perlu untuk setup.
- Kalau sesi ini sudah masuk `/teamwork`: pakai integrity `development`, lewati interview, jangan pindah ke `~/teamwork_projects/`, kerjakan di workspace ini, jangan spawn Critic/Challenger/Auditor, cukup smoke check, lalu berhenti.
- Jangan menulis test suite pytest yang lengkap, jangan verifikasi UI lewat browser, jangan training dataset penuh saat setup awal.
- Rencana cukup beberapa bullet. Jangan membuat dokumen rencana yang panjang.

Pembagian kerja yang disarankan:

1. `src/` + `config.yaml` (Worker A)
2. `app.py` (Worker B)
3. `notebooks/training.ipynb`, `README.md`, `requirements.txt`, `.gitignore`, skrip data, placeholder `models/` (Worker C)

## Invarian Arsitektur (jangan dilanggar)

- **Dataset Acuan:** Kaggle "Garbage Images Dataset" (slug: `zlatan599/garbage-dataset-classification`, Lisensi MIT) dengan subfolder: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- **Ekstraksi Fitur:** MobileNetV2 pretrained ImageNet (pooling='avg' -> 1280 dimensi float). Fitur handcrafted (HSV histogram + statistik + GLCM ~26 dimensi) disediakan sebagai baseline pembanding.
- **Data Splitting & Anti-Leakage:** Wajib stratified train/test split (80:20) **SEBELUM** melakukan `StandardScaler.fit`, `PCA.fit`, dan `FCM.fit`. `StandardScaler` dan `PCA` HANYA di-fit pada data train.
- **FCM NumPy Manual:** FCM ditulis manual di `src/fcm.py` memakai NumPy murni. Jangan menjadikan library FCM siap pakai sebagai implementasi utama. Menghasilkan pusat cluster $V$, keanggotaan $U$, riwayat fungsi objektif $J$, FPC, dan Partition Entropy (PE).
- **Membership Data Uji / Baru:** Derajat keanggotaan data uji dihitung menggunakan fungsi `fcm_predict_membership(X, centers, m)` berdasarkan jarak ke pusat cluster hasil training (bukan fit ulang FCM).
- **KNN Hybrid (+ mu):** KNN dilatih pada ruang gabungan `[X_pca, mu * mu_weight]` dengan label kelas aktual.
- Notebook Colab dan Streamlit **wajib import dari `src/`**. Jangan menyalin ulang logika ekstraksi fitur di notebook atau di `app.py`.
- Prediksi foto baru memakai artefak di `models/` (`scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, `meta.json`). Tombol **latih ulang** di GUI hanya menyimpan hasil di `st.session_state` dan **tidak menimpa** `models/`.
- Framing masalah: pengelompokan foto sampah yang belum berlabel di TPS/Bank Sampah. Sifat keanggotaan kontinu FCM mengatasi ambiguitas material.
- Teks UI wajib Bahasa Indonesia. Identifier kode, nama file, dan kunci `config.yaml` tetap bahasa Inggris.
- Setiap tahap algoritma harus bisa dijelaskan. Siapkan bagian contoh perhitungan manual FCM (5 foto × 2 komponen PCA, `c=2`, 1–2 iterasi, plus Partition Entropy) di notebook dan README.
- Pin versi `scikit-learn==1.4.2` dan `tensorflow>=2.15.0` di `requirements.txt`, dan pastikan versi sama persis antara Google Colab dan lokal laptop sebelum menyimpan `.pkl`.
