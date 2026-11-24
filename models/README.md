# Direktori Artefak Model (`models/`)

Direktori ini berisi artefak biner dan metadata hasil proses pelatihan model dari notebook (`notebooks/training.ipynb`) atau pipeline training.

---

## 1. Daftar Artefak yang Dihasilkan

Setelah proses training selesai dan fungsi `save_models()` dijalankan, direktori ini akan berisi 5 file utama:

1. **`scaler.pkl`**
   - Objek `sklearn.preprocessing.StandardScaler` yang di-fit **hanya pada data training** (fitur MobileNetV2 1280-dimensi atau Handcrafted 26-dimensi).
   - Digunakan untuk menstandarisasi fitur citra baru sebelum proyeksi PCA.

2. **`pca.pkl`**
   - Objek `sklearn.decomposition.PCA` yang di-fit **hanya pada data training** yang telah distandarisasi (dengan ambang variansi kumulatif 90%, dibatasi maksimum 50 komponen).
   - Digunakan untuk mereduksi dimensi fitur citra baru ke ruang laten PCA.

3. **`fcm_centers.npy`**
   - File biner NumPy array (berdimensi `c x n_components`) berisi koordinat pusat cluster FCM di ruang PCA.
   - Digunakan oleh fungsi `fcm_predict_membership()` untuk menghitung derajat keanggotaan fuzzy $\mu_{ij}$ untuk citra baru berdasarkan jarak ke pusat cluster tanpa perlu menjalankan iterasi FCM ulang.

4. **`knn.pkl`**
   - Objek `sklearn.neighbors.KNeighborsClassifier` yang dilatih pada ruang fitur gabungan:
     $$\mathbf{X}_{KNN} = [\mathbf{X}_{PCA}, \boldsymbol{\mu} \times \text{mu\_weight}]$$
     dengan label kelas sampah aktual dari folder dataset.
   - Menggabungkan representasi geometri laten (PCA) dan derajat keanggotaan fuzzy ($\mu$) untuk klasifikasi yang lebih robust terhadap ambiguitas material.

5. **`meta.json`**
   - File metadata konfigurasi model dalam format JSON yang menyimpan:
     * `class_names`: Daftar nama kelas kategori sampah (dinamis dari subfolder).
     * `feature_extractor`: Jenis ekstraktor fitur yang digunakan (`mobilenetv2` atau `handcrafted`).
     * `mu_weight`: Bobot pengali derajat keanggotaan fuzzy pada input KNN (default: 1.0).
     * `n_components`: Jumlah komponen PCA yang terpilih.
     * `variance_ratio`: Nilai total explained variance kumulatif.
     * `library_versions`: Versi library saat training (`scikit-learn`, `tensorflow`, `numpy`, `joblib`).

---

## 2. Alur Inferensi pada Citra Baru

Ketika citra baru diuji melalui aplikasi Streamlit (`app.py`):
1. Citra baru diekstraksi menggunakan ekstraktor yang sama (`MobileNetV2` atau `Handcrafted`).
2. Fitur dinormalisasi menggunakan `scaler.pkl`.
3. Fitur diproyeksikan ke ruang laten menggunakan `pca.pkl`.
4. Derajat keanggotaan fuzzy $\mu$ ke tiap cluster FCM dihitung menggunakan `fcm_predict_membership(X_pca, fcm_centers, m)`.
5. Vektor fitur PCA dan $\mu \times \text{mu\_weight}$ digabungkan secara horizontal:
   $$\mathbf{x}_{\text{eval}} = [\mathbf{x}_{\text{pca}}, \boldsymbol{\mu} \times \text{mu\_weight}]$$
6. `knn.pkl` memprediksi kelas akhir sampah dan probabilitasnya.

---

## 3. Peringatan Penting: Keselarasan Versi Dependensi

> **PERINGATAN SERIALISASI JOBLIB & TENSORFLOW:**
> File `.pkl` yang disimpan menggunakan `joblib` sangat sensitif terhadap versi `scikit-learn`.
> Pastikan versi `scikit-learn` pada **Google Colab** dan di **komputer lokal** sama persis (telah di-pin pada `scikit-learn==1.4.2` di `requirements.txt`).
> Perbedaan versi minor dapat mengakibatkan error deserialisasi (`InconsistentVersionWarning`, `AttributeError`, atau traceback crash).

---

## 4. Integrasi dengan GUI Streamlit

- Aplikasi Streamlit (`app.py`) memanggil fungsi `load_models(models_dir='models')` dari `src/pipeline.py` saat startup.
- **Handling Kondisi Kosong:** Jika file artefak di folder ini belum ada (misalnya saat repository baru di-clone), aplikasi GUI **tidak akan crash**. Aplikasi akan menampilkan pesan informatif ramah yang menyarankan pengguna untuk menjalankan pelatihan via Google Colab atau melakukan pelatihan cepat dalam memori pada tab *Proses*.
- **Keamanan:** Tombol "Latih Ulang" pada GUI Streamlit hanya menyimpan model di `st.session_state` memori browser dan **TIDAK** menimpa file di dalam direktori `models/` ini.
