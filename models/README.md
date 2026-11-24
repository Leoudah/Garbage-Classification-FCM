# Direktori Artefak Model (`models/`)

Direktori ini berisi artefak biner dan metadata hasil proses pelatihan model dari notebook (`notebooks/training.ipynb`) atau pipeline training.

---

## 1. Daftar Artefak yang Dihasilkan

Setelah proses training selesai dan fungsi `save_models()` dijalankan, direktori ini akan berisi 5 file utama:

1. **`scaler.pkl`**
   - Objek `sklearn.preprocessing.StandardScaler` yang di-fit **hanya pada data training** (fitur MobileNetV3-Small 576-dimensi atau Handcrafted).
   - Digunakan untuk menstandarisasi fitur citra baru sebelum proyeksi PCA.

2. **`pca.pkl`**
   - Objek `sklearn.decomposition.PCA` yang di-fit **hanya pada data training** yang telah distandarisasi (dengan 50 komponen atau ambang variansi kumulatif 95%).
   - Digunakan untuk mereduksi dimensi fitur citra baru ke ruang laten PCA.

3. **`fcm_centers.npy`**
   - File biner NumPy array (berdimensi `c x n_components`) berisi koordinat pusat cluster FCM di ruang PCA yang dihasilkan oleh `skfuzzy.cluster.cmeans`.
   - Digunakan oleh fungsi `fcm_predict_membership()` via `skfuzzy.cluster.cmeans_predict` untuk menghitung derajat keanggotaan fuzzy $\mu_{ij}$ untuk citra baru berdasarkan jarak ke pusat cluster tanpa perlu menjalankan iterasi FCM ulang.

4. **`knn.pkl`**
   - Objek `sklearn.neighbors.KNeighborsClassifier` yang dilatih pada ruang fitur gabungan:
     $$\mathbf{X}_{KNN} = [\mathbf{X}_{PCA} \parallel \alpha \cdot \boldsymbol{\mu}]$$
     dengan label kelas sampah aktual dari folder dataset.
   - Menggabungkan representasi geometri laten (PCA) dan derajat keanggotaan fuzzy ($\mu$) dengan bobot scaling $\alpha$ (default 1.0) untuk klasifikasi yang lebih robust terhadap ambiguitas material.

5. **`meta.json`**
   - File metadata konfigurasi model dalam format JSON yang menyimpan:
     * `class_names`: Daftar nama kelas kategori sampah.
     * `feature_extractor`: Jenis ekstraktor fitur (`mobilenetv3_small` atau `handcrafted`).
     * `alpha`: Bobot pengali derajat keanggotaan fuzzy pada input KNN (default: 1.0).
     * `n_components`: Jumlah komponen PCA yang terpilih.
     * `variance_ratio`: Nilai total explained variance kumulatif.
     * `scenarios`: Ringkasan metrik evaluasi 4 skenario pengujian (1a, 1b, 2a, 2b).
     * `library_versions`: Versi library saat training (`scikit-learn`, `scikit-fuzzy`, `tensorflow`, `numpy`, `joblib`).

---

## 2. Alur Inferensi pada Citra Baru

Ketika citra baru diuji melalui aplikasi Streamlit (`app.py`):
1. Citra baru di-resize ke $224 \times 224$ dan diekstraksi menggunakan `MobileNetV3-Small` (atau `Handcrafted`).
2. Fitur dinormalisasi menggunakan `scaler.pkl`.
3. Fitur diproyeksikan ke ruang laten menggunakan `pca.pkl`.
4. Derajat keanggotaan fuzzy $\mu$ ke tiap cluster FCM dihitung menggunakan `fcm_predict_membership(X_pca, fcm_centers, m)`.
5. Vektor fitur PCA dan $\alpha \cdot \boldsymbol{\mu}$ digabungkan secara horizontal:
   $$\mathbf{x}_{\text{eval}} = [\mathbf{x}_{\text{pca}} \parallel \alpha \cdot \boldsymbol{\mu}]$$
6. `knn.pkl` memprediksi kelas akhir sampah dan probabilitasnya.

---

## 3. Integrasi dengan GUI Streamlit

- Aplikasi Streamlit (`app.py`) memanggil fungsi `load_models(models_dir='models')` dari `src/pipeline.py` saat startup.
- **Handling Kondisi Kosong:** Jika file artefak di folder ini belum ada, aplikasi GUI menampilkan pesan informatif ramah yang menyarankan pengguna untuk menjalankan pelatihan via Google Colab atau melakukan pelatihan cepat dalam memori pada tab *Proses*.
- **Keamanan:** Tombol "Latih Ulang" pada GUI Streamlit hanya menyimpan model di `st.session_state` memori browser dan **TIDAK** menimpa file di dalam direktori `models/` ini.
