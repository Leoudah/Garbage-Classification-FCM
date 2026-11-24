# Spesifikasi — Final Project TLF: Clusterisasi & Klasifikasi Foto Sampah

Mata kuliah Teknologi Logika Fuzzy (TLF). Aplikasi wajib ber-GUI (Streamlit). Metode fuzzy (FCM) dan supervised non-fuzzy (KNN) **terintegrasi secara erat** melalui representasi gabungan fitur laten PCA dan derajat keanggotaan fuzzy $\mu$.

---

## 1. Masalah yang Diselesaikan

Mengelompokkan dan mengklasifikasikan foto sampah yang di lapangan (TPS atau Bank Sampah) biasanya **belum berlabel (*unlabeled*)**. FCM mengelompokkan secara *unsupervised*; sifat keanggotaan kontinu fuzzy-nya sangat cocok karena sampah di dunia nyata seringkali ambigu:
- Kardus lembab/kotor memiliki kemiripan warna dan tekstur dengan lembaran kertas.
- Plastik transparan / botol bening memiliki kemiripan pantulan cahaya dengan pecahan kaca.
- Kaleng logam bekas memiliki tekstur atau warna mirip plastik/kardus.

Dataset acuan: **Garbage Images Dataset** di Kaggle (slug: `zlatan599/garbage-dataset-classification`, Lisensi MIT), dengan kategori:
`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

Prinsip evaluasi & KNN:
- FCM belajar murni secara *unsupervised* pada fitur PCA tanpa melihat label nama folder.
- Label kelas hanya digunakan untuk melatih KNN (yang memetakan kombinasi `[fitur_pca, mu * mu_weight]` ke kelas nyata) dan menghitung metrik evaluasi klasifikasi (Akurasi, Precision macro, Recall macro, F1 macro, Confusion Matrix).

---

## 2. Alur Metode

1. **Ekstraksi Fitur:**
   - **Fitur Utama (Deep Features):** MobileNetV2 pretrained ImageNet (pooling='avg') menghasilkan 1280 dimensi representasi semantik tinggi per citra.
   - **Fitur Pembanding (Handcrafted):** HSV Histogram (16) + Statistik Warna (6) + GLCM Tekstur (4) = 26 dimensi, untuk studi komparasi.
2. **Stratified Split & Preprocessing (Anti Data Leakage):**
   - Data dibagi menjadi training set (80%) dan testing set (20%) secara stratified berdasarkan kelas.
   - `StandardScaler` di-fit **hanya** pada data train, lalu digunakan untuk mentransformasi train dan test.
3. **Reduksi Dimensi PCA:**
   - `PCA` di-fit **hanya** pada data train yang telah diskalakan dengan ambang variansi kumulatif 90% (dibatasi maksimum 50 komponen).
   - Menghasilkan representasi kompak dan komponen PC1 vs PC2 untuk visualisasi scatter plot di GUI.
4. **FCM (Fuzzy C-Means Manual via NumPy):**
   - Di-fit **hanya** pada data train ruang PCA secara murni tanpa library eksternal.
   - Menghitung matriks keanggotaan $U_{train}$, pusat cluster $V$, riwayat fungsi objektif $J$, Fuzzy Partition Coefficient (FPC), dan Partition Entropy (PE).
   - Derajat keanggotaan data uji dihitung melalui `fcm_predict_membership(X_test_pca, V, m)` berdasarkan jarak Euclidean ke pusat cluster.
5. **KNN Classifier Terpadu (+ mu):**
   - Vektor input KNN dibentuk dari penggabungan fitur PCA dan derajat keanggotaan fuzzy:
     $$\mathbf{X}_{KNN} = [\mathbf{X}_{PCA}, \boldsymbol{\mu} \times \text{mu\_weight}]$$
   - Dilatih dengan label kelas aktual. Citra baru diprediksi secara instan (lazy learner) tanpa menjalankan ulang FCM dari nol.
6. **Evaluasi & Studi Komparasi:**
   - Evaluasi FCM: Silhouette score di ruang PCA, FPC, dan Partition Entropy (PE).
   - Evaluasi KNN: Akurasi, Precision macro, Recall macro, F1-score macro, Confusion Matrix.
   - **Dua Komponen Perbandingan:**
     a. *KNN Biasa* (hanya fitur PCA) vs *KNN + mu FCM* (fitur PCA + derajat keanggotaan fuzzy).
     b. *MobileNetV2* (1280 dimensi) vs *Handcrafted HSV+GLCM* (26 dimensi).

---

## 3. Struktur Direktori Repository

```
Garbage-Classification-FCM/
├── AGENTS.md                  # Aturan kerja & panduan peran agen
├── SPESIFIKASI-PROYEK.md      # Spesifikasi lengkap proyek
├── .agents/                   # Konfigurasi skill dan rule agen
├── app.py                     # Antarmuka web Streamlit
├── config.yaml                # Konfigurasi parameter model & paths
├── requirements.txt           # Dependensi pustaka (sklearn==1.4.2, tensorflow>=2.15.0)
├── README.md                  # Dokumentasi proyek & panduan pemakaian
├── .gitignore
├── src/
│   ├── __init__.py            # Ekspos API publik
│   ├── features.py            # MobileNetV2 (1280D) & Handcrafted HSV+GLCM (26D)
│   ├── fcm.py                 # Algoritma FCM manual (NumPy), FPC, PE, fcm_predict_membership
│   └── pipeline.py            # Pipeline fit, transform, predict, compare, load/save models
├── notebooks/
│   └── training.ipynb         # Notebook Google Colab: unduh data, manual FCM, train, export
├── models/                    # Artefak model hasil pelatihan
│   ├── README.md
│   ├── .gitkeep
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── fcm_centers.npy
│   ├── knn.pkl
│   └── meta.json
└── data/                      # Data citra sampah
    ├── README.md
    └── download_dataset.py    # Skrip unduh Garbage Images Dataset via Kaggle API
```

---

## 4. Halaman Antarmuka Streamlit (`app.py`)

Bahasa antarmuka wajib Bahasa Indonesia, validasi input citra (JPG, PNG, maks 10MB):
1. **Beranda:** Framing masalah di TPS/Bank Sampah, sifat keanggotaan kontinu FCM dalam menangani ambiguitas material sampah, dan alur metode 4 tahap.
2. **Upload:** Unggah foto uji untuk inferensi dengan preview thumbnail, opsi tombol sampel cepat, dan opsi unggah dataset ZIP.
3. **Proses:** Laboratorium latih ulang interaktif pada subset data (10–20 sampel per kelas) dengan slider parameter ($c, m, k, \text{mu\_weight}, n\_components, \text{test\_size}$). Fitur diekstrak dan di-cache di `st.session_state`. Menampilkan diagram explained variance PCA, kurva fungsi objektif $J$ FCM, dan pelatihan KNN. *(Tidak menimpa folder `models/`)*.
4. **Hasil:** Scatter plot ruang proyeksi PC1 vs PC2 (sebaran data, pusat cluster FCM, posisi foto baru), tabel derajat keanggotaan fuzzy $\mu_{ij}$ beserta visual progress bar, dan interpretasi ramah.
5. **Evaluasi:** Metrik evaluasi FCM (FPC, PE, Silhouette), metrik KNN (Accuracy, Precision, Recall, F1, Confusion Matrix), serta **dua tabel perbandingan**:
   - KNN Biasa vs KNN + mu FCM
   - MobileNetV2 vs Handcrafted (HSV + GLCM)
6. **Petunjuk / Info Sistem:** Panduan teknis penggunaan, tabel parameter default, versi pustaka lingkungan, dan keterbatasan sistem.

---

## 5. Notebook Google Colab (`notebooks/training.ipynb`)

1. Lingkungan: Deteksi Colab, mount Drive / clone git, setting `sys.path`.
2. Pengecekan versi `scikit-learn==1.4.2` dan `tensorflow>=2.15.0`.
3. Panduan unduh Garbage Images Dataset via Kaggle API (`zlatan599/garbage-dataset-classification`).
4. Import modul resmi dari `src/` (tidak menulis ulang logika di sel notebook).
5. **Bagian Simulasi Perhitungan Manual FCM:** 5 data $\times$ 2 komponen PCA, $c=2$, $m=2.0$, 2 iterasi lengkap, dilengkapi rumus matematis, matriks inisialisasi $U$, perhitungan pusat cluster $V$, matriks jarak $D$, fungsi objektif $J$, pembaruan $U$, serta perhitungan numerik **Partition Entropy (PE)**.
6. Pipeline training terpadu: stratified split $\rightarrow$ ekstraksi fitur $\rightarrow$ scaler (train only) $\rightarrow$ PCA (train only, max 50) $\rightarrow$ FCM manual $\rightarrow$ `fcm_predict_membership` $\rightarrow$ KNN input `[PCA, mu * mu_weight]`.
7. Evaluasi lengkap dan dua sel studi komparasi (KNN vs KNN+mu; MobileNetV2 vs Handcrafted).
8. Simpan artefak model ke `models/` (`scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, `meta.json`) dan verifikasi pemuatan kembali melalui `load_models()`.
