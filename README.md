# Final Project TLF: Klasifikasi & Clusterisasi Foto Sampah Terpadu
> **Mata Kuliah: Teknologi Logika Fuzzy (TLF)**  
> Arsitektur Terpadu: **MobileNetV2 (1280D) $\rightarrow$ StandardScaler $\rightarrow$ PCA $\rightarrow$ Fuzzy C-Means (NumPy Manual) $\rightarrow$ KNN (+ $\mu$ FCM)** dengan Antarmuka Interaktif **Streamlit**.

---

## 1. Latar Belakang & Framing Masalah

Pada pengelolaan sampah perkotaan di Tempat Pengolahan Sampah (TPS) maupun Bank Sampah, citra foto sampah yang dikumpulkan di lapangan pada umumnya **belum memiliki label (*unlabeled*)**. Selain itu, karakteristik fisik sampah di dunia nyata memiliki batas-batas material yang sangat ambigu:
- Kardus yang basah atau kotor memiliki warna kecokelatan dan tekstur serat yang menyerupai kertas biasa.
- Plastik transparan / botol bening memiliki kemiripan pantulan cahaya dan nilai intensitas dengan pecahan kaca.
- Kaleng logam bekas yang berkarat memiliki kemiripan warna dan tekstur dengan material kardus.

Pendekatan *crisp clustering* (seperti K-Means biasa) memaksakan setiap data hanya menjadi anggota tunggal (100% atau 0%), yang tidak realistis untuk objek ambigu. **Fuzzy C-Means (FCM)** dipilih karena memungkinkan setiap data memiliki **derajat keanggotaan kontinu ($\mu_{ij} \in [0, 1]$)** ke beberapa cluster sekaligus dengan batasan $\sum_j \mu_{ij} = 1$.

### Dataset Acuan: Garbage Images Dataset
Dataset yang digunakan adalah **Garbage Images Dataset** dari Kaggle (Lisensi MIT):
- **Kaggle Dataset:** [zlatan599/garbage-dataset-classification](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification)
- **Kategori / Kelas:**
  1. `cardboard` (kardus)
  2. `glass` (kaca)
  3. `metal` (logam)
  4. `paper` (kertas)
  5. `plastic` (plastik)
  6. `trash` (sampah residu/lainnya)

> **Prinsip Evaluasi & Pembelajaran:**
> 1. Algoritma **FCM** belajar secara murni **unsupervised** pada ruang fitur PCA tanpa pernah membaca label kelas nama folder.
> 2. Label kelas asli dari subfolder hanya digunakan untuk melatih classifier **KNN** pemetaan dan menghitung metrik evaluasi klasifikasi (Akurasi, Precision macro, Recall macro, F1-macro, dan Confusion Matrix).

---

## 2. Arsitektur & Alur Metode

Alur pengolahan dirancang secara sistematis dengan pencegahan kebocoran data (*anti-data leakage*):

```
+--------------------+
| Foto Sampah        |
| (Input Citra)      |
+--------------------+
          |
          v
+--------------------+     +-----------------------------------------------+
| Ekstraksi Fitur    | --> | MobileNetV2 (Pretrained ImageNet, 1280 Dim)   |
|                    |     | *Pembanding: Handcrafted HSV+GLCM (~26 Dim)   |
+--------------------+     +-----------------------------------------------+
          |
          v
+--------------------+
| Stratified Split   | (80% Train, 20% Test)
+--------------------+
          |
          +-----------------------+-----------------------+
          | (Data Train)                                  | (Data Test)
          v                                               v
+--------------------+                          +--------------------+
| StandardScaler.fit |                          | Scaler.transform   |
+--------------------+                          +--------------------+
          |                                               |
          v                                               v
+--------------------+                          +--------------------+
| PCA.fit (Var 90%)  |                          | PCA.transform      |
+--------------------+                          +--------------------+
          |                                               |
          v                                               |
+--------------------+                                    |
| FCM Manual (NumPy) | --> Pusat Cluster V                |
| Menghasilkan U_tr  |              |                     |
+--------------------+              v                     v
          |                   +--------------------------------+
          |                   | fcm_predict_membership(X_test) |
          |                   | Menghasilkan U_test            |
          |                   +--------------------------------+
          v                                               v
+------------------------------------+          +------------------------------------+
| Input KNN Train:                   |          | Input KNN Test:                    |
| [X_train_pca, mu_tr * mu_weight]   |          | [X_test_pca, mu_te * mu_weight]    |
+------------------------------------+          +------------------------------------+
          |                                               |
          v                                               v
+--------------------+                          +--------------------+
| KNN Classifier.fit | -----------------------> | Prediksi & Evaluasi|
+--------------------+                          +--------------------+
```

1. **Ekstraksi Fitur:**
   - **Utama (MobileNetV2):** Model deep representation CNN pretrained ImageNet (`pooling='avg'`) menghasilkan 1280 fitur bersemantik tinggi.
   - **Pembanding (Handcrafted):** HSV histogram 16 bin + statistik warna 6 fitur + tekstur GLCM 4 fitur = 26 dimensi.
2. **Stratified Split & Normalisasi:**
   - Pemisahan data training (80%) dan testing (20%) dilakukan **sebelum** fitting.
   - `StandardScaler` di-fit hanya pada data training.
3. **Reduksi Dimensi PCA:**
   - PCA di-fit hanya pada data training dengan target variansi kumulatif 90% (dibatasi maksimum 50 komponen).
4. **Fuzzy C-Means (NumPy Manual):**
   - Di-fit pada data training ruang PCA secara murni tanpa library eksternal.
   - Menghasilkan matriks keanggotaan $U_{train}$, koordinat pusat cluster $V$, kurva fungsi objektif $J$, Fuzzy Partition Coefficient (FPC), dan Partition Entropy (PE).
   - Derajat keanggotaan data uji dihitung melalui fungsi `fcm_predict_membership()` berdasarkan jarak Euclidean ke pusat cluster $V$.
5. **KNN Classifier Hybrid (+ $\mu$ FCM):**
   - Vektor input KNN memadukan koordinat laten PCA dan derajat keanggotaan fuzzy:
     $$\mathbf{X}_{KNN} = [\mathbf{X}_{PCA}, \boldsymbol{\mu} \times \text{mu\_weight}]$$
   - Prediksi citra baru berlangsung instan (*lazy learner*) tanpa perlu iterasi FCM ulang.
6. **Dua Komponen Studi Komparasi:**
   - **Studi 1:** *KNN Biasa* (fitur PCA saja) vs *KNN + $\mu$ FCM* (fitur PCA + derajat keanggotaan fuzzy).
   - **Studi 2:** *MobileNetV2* (1280 dimensi) vs *Handcrafted HSV+GLCM* (26 dimensi).

---

## 3. Struktur Direktori Repository

```
Garbage-Classification-FCM/
├── app.py                      # Aplikasi Web GUI Streamlit (6 menu navigasi)
├── config.yaml                 # Konfigurasi parameter model, ekstraktor, & paths
├── requirements.txt            # Dependensi pustaka (sklearn==1.4.2, tensorflow>=2.15.0)
├── README.md                   # Dokumentasi lengkap proyek
├── .gitignore                  # Pengecualian dataset citra, venv, & cache
├── src/                        # Modul algoritma inti
│   ├── __init__.py             # API publik paket
│   ├── features.py             # Ekstraksi MobileNetV2 (1280D) & Handcrafted (26D)
│   ├── fcm.py                  # FCM NumPy manual, FPC, PE, fcm_predict_membership
│   └── pipeline.py             # Pipeline fit, transform, predict, compare, save/load
├── notebooks/
│   └── training.ipynb          # Notebook Google Colab (unduh data, manual FCM, train, export)
├── data/                       # Direktori dataset citra
│   ├── README.md               # Panduan struktur dan unduh Garbage Images Dataset
│   └── download_dataset.py     # Skrip unduh otomatis via Kaggle API
└── models/                     # Direktori artefak model (hasil training)
    ├── README.md               # Dokumentasi artefak model
    ├── .gitkeep                # Pelacak git
    ├── scaler.pkl              # Objek StandardScaler
    ├── pca.pkl                 # Objek PCA
    ├── fcm_centers.npy         # Koordinat pusat cluster FCM (NumPy array)
    ├── knn.pkl                 # Model KNeighborsClassifier terlatih
    └── meta.json               # Metadata konfigurasi & versi pustaka
```

---

## 4. Panduan Instalasi & Menjalankan Lokal

### Prasyarat
- Python 3.10, 3.11, atau 3.12 terpasang.
- Koneksi internet untuk unduhan awal model pretrained MobileNetV2.

### Langkah Instalasi
1. **Clone repository atau buka direktori proyek:**
   ```bash
   git clone https://github.com/<username>/Garbage-Classification-FCM.git
   cd Garbage-Classification-FCM
   ```

2. **Buat dan aktifkan lingkungan virtual:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Pasang dependensi pustaka:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi GUI Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di browser Anda pada alamat `http://localhost:8501`.

---

## 5. Fitur Antarmuka Streamlit (`app.py`)

Aplikasi dilengkapi 6 halaman navigasi interaktif berbahasa Indonesia:
1. **Beranda:** Framing masalah klasifikasi sampah ambigu di TPS/Bank Sampah, sifat keanggotaan kontinu FCM, dan alur metode 4 tahap.
2. **Upload:** Unggah foto uji (JPG/PNG, maks 10MB) dengan thumbnail preview, opsi tombol sampel cepat, dan opsi unggah dataset ZIP.
3. **Proses:** Laboratorium latih ulang interaktif pada subset data (10–20 sampel per kelas) dengan kontrol slider ($c, m, k, \text{mu\_weight}, n\_components, \text{test\_size}$). Fitur diekstrak dan di-cache di `st.session_state` sehingga pergantian parameter tidak memerlukan ekstraksi ulang. *(Tidak menimpa folder `models/`)*.
4. **Hasil:** Scatter plot ruang proyeksi PC1 vs PC2 (sebaran data, pusat cluster FCM, posisi foto baru), tabel derajat keanggotaan fuzzy $\mu_{ij}$ beserta visual bar, dan interpretasi prediksi ramah.
5. **Evaluasi:** Metrik evaluasi FCM (FPC, PE, Silhouette), metrik KNN (Accuracy, Precision, Recall, F1, Confusion Matrix), serta **dua tabel studi komparasi**:
   - Komparasi 1: KNN Biasa vs KNN + $\mu$ FCM
   - Komparasi 2: MobileNetV2 vs Handcrafted (HSV + GLCM)
6. **Petunjuk / Info Sistem:** Panduan teknis penggunaan, tabel parameter default, versi pustaka lingkungan, dan keterbatasan sistem.

> **Safety Fallback:** Jika folder `models/` belum memiliki file biner hasil training, aplikasi **tidak akan crash**. Aplikasi akan menampilkan pesan informatif ramah yang menyarankan pengguna melakukan latih ulang via menu *Proses* atau menjalankan Colab notebook.

---

## 6. Panduan Penggunaan di Google Colab (`notebooks/training.ipynb`)

1. Buka [Google Colab](https://colab.research.google.com).
2. Unggah file `notebooks/training.ipynb` atau buka langsung dari repository GitHub.
3. Jalankan sel secara berurutan:
   - Verifikasi keselarasan versi `scikit-learn==1.4.2` dan `tensorflow>=2.15.0`.
   - Unduh dataset Garbage Images Dataset via Kaggle API (`zlatan599/garbage-dataset-classification`).
   - Pelajari **Simulasi Perhitungan Manual FCM & Partition Entropy** (5 data $\times$ 2 komponen PCA).
   - Ekstrak fitur MobileNetV2 dan latih pipeline terpadu (`fit_pipeline`).
   - Analisis grafik fungsi objektif $J$, scatter plot PC1 vs PC2, dan confusion matrix data uji.
   - Jalankan dua sel studi komparasi (KNN vs KNN+$\mu$, dan MobileNetV2 vs Handcrafted).
   - Simpan artefak model ke `models/` (`scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, `meta.json`).
4. Unduh isi folder `models/` dari Colab ke laptop Anda untuk langsung digunakan oleh aplikasi Streamlit lokal.

---

## 7. Penjelasan Contoh Perhitungan Manual FCM & Entropy

Sesuai ketentuan akademik TLF, setiap langkah algoritma harus dapat dibuktikan secara matematis. Pada `notebooks/training.ipynb` disediakan simulasi manual FCM dengan parameter:
- **Jumlah Data ($N$):** 5 sampel data 2D ($PC_1, PC_2$):
  - $x_1 = [1.0, 1.5]$
  - $x_2 = [1.2, 1.8]$
  - $x_3 = [0.8, 1.2]$
  - $x_4 = [4.5, 4.0]$
  - $x_5 = [5.0, 4.5]$
- **Jumlah Cluster ($c$):** 2 cluster ($v_1, v_2$).
- **Fuzzifier ($m$):** 2.0.

### Rumus Matematis:
1. **Pusat Cluster ($v_j$):**
   $$v_j = \frac{\sum_{i=1}^N (u_{ij})^m x_i}{\sum_{i=1}^N (u_{ij})^m}$$
2. **Jarak Euclidean ($d_{ij}$):**
   $$d_{ij} = \|x_i - v_j\|_2 = \sqrt{\sum_{k=1}^p (x_{ik} - v_{jk})^2}$$
3. **Fungsi Objektif ($J$):**
   $$J = \sum_{i=1}^N \sum_{j=1}^c (u_{ij})^m (d_{ij})^2$$
4. **Pembaruan Matriks Keanggotaan ($u_{ij}$):**
   $$u_{ij} = \frac{1}{\sum_{k=1}^c \left(\frac{d_{ij}}{d_{ik}}\right)^{\frac{2}{m-1}}}$$
5. **Fuzzy Partition Coefficient (FPC):**
   $$\text{FPC} = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^c (u_{ij})^2$$
6. **Partition Entropy (PE):**
   $$\text{PE} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^c u_{ij} \ln(u_{ij}) \quad (\text{untuk } u_{ij} > 0)$$

---

## 8. Catatan Penting Mengenai Versi Pustaka

> **PERINGATAN SERIALISASI MODEL:**
> Artefak model diserialisasi menggunakan `joblib`.
> Versi `scikit-learn` pada Google Colab dan di komputer lokal **WAJIB SAMA** (`scikit-learn==1.4.2`).
> Pastikan juga versi `tensorflow` kompatibel ($\ge 2.15.0$) agar ekstraksi MobileNetV2 menghasilkan bobot fitur yang identik.
