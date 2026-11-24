# Final Project TLF: Klasifikasi & Clusterisasi Foto Sampah Terpadu
> **Mata Kuliah: Teknologi Logika Fuzzy (TLF)**  
> Implementasi Full Library: **MobileNetV3-Small (576D) $\rightarrow$ StandardScaler $\rightarrow$ PCA $\rightarrow$ scikit-fuzzy FCM $\rightarrow$ KNN (+ $\alpha \cdot \mu$)** dengan Antarmuka Interaktif **Streamlit**.  
> **Repository GitHub:** [https://github.com/YasMbul/Garbage-Classification-FCM.git](https://github.com/YasMbul/Garbage-Classification-FCM.git)

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
+--------------------+     +-------------------------------------------------+
| Ekstraksi Fitur    | --> | MobileNetV3-Small (Pretrained ImageNet, 576 Dim)|
|                    |     | *Pembanding: Handcrafted HSV+GLCM (~26 Dim)     |
+--------------------+     +-------------------------------------------------+
          |
          v
+--------------------+
| Stratified Split   | (80% Train, 20% Test, seed=42)
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
| PCA.fit (50 Komp)  |                          | PCA.transform      |
+--------------------+                          +--------------------+
          |                                               |
          v                                               |
+--------------------+                                    |
| skfuzzy FCM (cmeans| --> Pusat Cluster V                |
| Menghasilkan U_tr  |              |                     |
+--------------------+              v                     v
          |                   +------------------------------------+
          |                   | cmeans_predict(X_test_pca.T, V)    |
          |                   | Menghasilkan U_test                |
          |                   +------------------------------------+
          v                                               v
+------------------------------------+          +------------------------------------+
| Input KNN Train:                   |          | Input KNN Test:                    |
| [X_train_pca || alpha * mu_tr]     |          | [X_test_pca || alpha * mu_te]      |
+------------------------------------+          +------------------------------------+
          |                                               |
          v                                               v
+--------------------+                          +--------------------+
| KNN Classifier.fit | -----------------------> | Prediksi & Evaluasi|
+--------------------+                          +--------------------+
```

1. **Ekstraksi Fitur:**
   - **Utama (MobileNetV3-Small):** Model CNN ringan `MobileNetV3Small` pretrained ImageNet (`pooling='avg'`, `include_preprocessing=True`) menghasilkan 576 fitur float bersemantik tinggi dari citra $224 \times 224$ pada rentang piksel $[0, 255]$.
   - **Pembanding (Handcrafted):** HSV histogram 16 bin + statistik warna 6 fitur + tekstur GLCM 4 fitur = 26 dimensi.
2. **Sampling & Stratified Split:**
   - Random sampling 1.000 citra per kelas (total 6.000 citra) dengan `random_state=42`.
   - Pemisahan data training (80% = 4.800 sampel) dan testing (20% = 1.200 sampel) dilakukan **sebelum** fitting.
3. **Standarisasi & Reduksi Dimensi PCA:**
   - `StandardScaler` di-fit hanya pada data training.
   - `PCA` (50 komponen) di-fit hanya pada data training terstandarisasi.
4. **Fuzzy C-Means (scikit-fuzzy):**
   - Menggunakan `skfuzzy.cluster.cmeans` pada data latih di ruang PCA.
   - Menghasilkan pusat cluster $V$, kurva fungsi objektif $P$, Fuzzy Partition Coefficient (FPC), Partition Entropy (PE), dan Silhouette score.
   - Derajat keanggotaan data uji dihitung melalui `skfuzzy.cluster.cmeans_predict()`.
5. **KNN Classifier Hybrid (+ $\alpha \cdot \mu$ FCM):**
   - Vektor input KNN memadukan koordinat laten PCA dan derajat keanggotaan fuzzy dengan bobot $\alpha$ (default 1.0):
     $$\mathbf{X}_{KNN} = [\mathbf{X}_{PCA} \parallel \alpha \cdot \boldsymbol{\mu}]$$
   - Prediksi citra baru berlangsung instan (*lazy learner*) tanpa perlu iterasi FCM ulang.
6. **Evaluasi 4 Skenario Pengujian:**
   - **Skenario 1a:** MobileNetV3-Small + PCA + (FCM + KNN) $\rightarrow$ *Model Usulan*
   - **Skenario 1b:** MobileNetV3-Small + PCA + KNN
   - **Skenario 2a:** (Histogram HSV + GLCM) + PCA + (FCM + KNN)
   - **Skenario 2b:** (Histogram HSV + GLCM) + PCA + KNN

---

## 3. Struktur Direktori Repository

```
Garbage-Classification-FCM/
├── app.py                      # Aplikasi Web GUI Streamlit (6 menu navigasi)
├── config.yaml                 # Konfigurasi parameter model, ekstraktor, & paths
├── requirements.txt            # Dependensi pustaka Python
├── README.md                   # Dokumentasi lengkap proyek
├── .gitignore                  # Pengecualian dataset citra, venv, & cache
├── src/                        # Modul algoritma inti
│   ├── __init__.py             # API publik paket
│   ├── features.py             # Ekstraksi MobileNetV3-Small (576D) & Handcrafted (26D)
│   ├── fcm.py                  # Wrapper scikit-fuzzy FCM, FPC, PE, Silhouette
│   └── pipeline.py             # Pipeline fit, transform, predict, compare, save/load
├── notebooks/
│   └── training.ipynb          # Notebook Google Colab (unduh data, manual FCM, train, 4 skenario)
├── data/                       # Direktori dataset citra
│   ├── README.md               # Panduan struktur dan unduh Garbage Images Dataset
│   └── download_dataset.py     # Skrip unduh otomatis via Kaggle API
└── models/                     # Direktori artefak model (hasil training)
    ├── README.md               # Dokumentasi artefak model
    ├── .gitkeep                # Pelacak git
    ├── scaler.pkl              # Objek StandardScaler
    ├── pca.pkl                 # Objek PCA (50 komponen)
    ├── fcm_centers.npy         # Koordinat pusat cluster FCM (NumPy array)
    ├── knn.pkl                 # Model KNeighborsClassifier terlatih
    └── meta.json               # Metadata konfigurasi, versi library, & hasil skenario
```

---

## 4. Panduan Instalasi & Menjalankan Lokal

### Prasyarat
- Python 3.10, 3.11, atau 3.12 terpasang.
- Koneksi internet untuk unduhan awal bobot pretrained MobileNetV3-Small.

### Langkah Instalasi
1. **Clone repository atau buka direktori proyek:**
   ```bash
   git clone https://github.com/YasMbul/Garbage-Classification-FCM.git
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
3. **Proses:** Laboratorium latih ulang interaktif pada subset data dengan kontrol slider ($c, m, k, \alpha, n\_components, \text{test\_size}$). Fitur diekstrak dan di-cache di `st.session_state` sehingga pergantian parameter tidak memerlukan ekstraksi ulang. *(Tidak menimpa folder `models/`)*.
4. **Hasil:** Scatter plot ruang proyeksi PC1 vs PC2 (sebaran data, pusat cluster FCM, posisi foto baru), tabel derajat keanggotaan fuzzy $\mu_{ij}$ beserta visual bar, kurva konvergensi nilai $P$, dan interpretasi prediksi ramah.
5. **Evaluasi:** Metrik evaluasi FCM (FPC, PE, Silhouette), metrik KNN (Accuracy, Precision, Recall, F1, Confusion Matrix), serta **tabel perbandingan 4 skenario pengujian**:
   - Skenario 1a: MobileNetV3-Small + PCA + (FCM + KNN)
   - Skenario 1b: MobileNetV3-Small + PCA + KNN
   - Skenario 2a: (Histogram HSV + GLCM) + PCA + (FCM + KNN)
   - Skenario 2b: (Histogram HSV + GLCM) + PCA + KNN
6. **Petunjuk / Info Sistem:** Penjelasan full library, rumus-rumus internal FCM ($V, d^2, P, \mu$, FPC, PE, Silhouette), integrasi $[\text{PCA} \parallel \alpha \cdot \mu]$, tabel parameter default, dan panduan penggunaan.

> **Safety Fallback:** Jika folder `models/` belum memiliki file biner hasil training, aplikasi **tidak akan crash**. Aplikasi akan menampilkan pesan informatif ramah yang menyarankan pengguna melakukan latih ulang via menu *Proses* atau menjalankan Colab notebook.

---

## 6. Panduan Penggunaan di Google Colab (`notebooks/training.ipynb`)

1. Buka [Google Colab](https://colab.research.google.com).
2. Clone repository:
   ```bash
   !git clone https://github.com/YasMbul/Garbage-Classification-FCM.git
   %cd Garbage-Classification-FCM
   !pip install -r requirements.txt
   ```
3. Jalankan sel secara berurutan:
   - Unduh dataset Garbage Images Dataset via Kaggle API (`zlatan599/garbage-dataset-classification`).
   - Pelajari **Simulasi Perhitungan Manual FCM & Entropy** (5 data sampel "Kilap" dan "Coklat").
   - Ekstrak fitur MobileNetV3-Small dan latih pipeline terpadu (`fit_pipeline`).
   - Analisis grafik fungsi objektif $P$, scatter plot PC1 vs PC2, dan confusion matrix data uji.
   - Evaluasi 4 skenario komparasi (1a, 1b, 2a, 2b).
   - Simpan artefak model ke `models/` (`scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, `meta.json`).
4. Unduh isi folder `models/` dari Colab ke laptop Anda untuk langsung digunakan oleh aplikasi Streamlit lokal.

---

## 7. Penjelasan Contoh Perhitungan Manual FCM: Kilap & Coklat

Sesuai dokumen konteks tugas TLF, perhitungan manual disimulasikan menggunakan 5 data sampel dengan 2 fitur ("Kilap", "Coklat"):
- **A** = $(7, 1)$ — Plastik
- **E** = $(6, 2)$ — Plastik
- **B** = $(9, 1)$ — Kaca
- **C** = $(1, 9)$ — Kardus
- **D** = $(2, 7)$ — Kertas

**Parameter:** $c = 2$, $w = 2$ ($m = 2$), toleransi $\xi = 0.001$, $\text{MaxIter} = 100$, $k = 3$.

### Rumus Matematis:
1. **Pusat Cluster ($V_j$):**
   $$V_j = \frac{\sum_{i=1}^N (u_{ij})^w x_i}{\sum_{i=1}^N (u_{ij})^w}, \quad j = 1, 2, \dots, c$$
2. **Kuadrat Jarak Euclidean ($d_{ij}^2$):**
   $$d_{ij}^2 = \|x_i - V_j\|^2 = \sum_{k=1}^p (x_{ik} - V_{jk})^2$$
3. **Fungsi Objektif ($P$):**
   $$P = \sum_{i=1}^N \sum_{j=1}^c (u_{ij})^w d_{ij}^2$$
4. **Pembaruan Matriks Keanggotaan ($u_{ij}$):**
   $$u_{ij} = \frac{1}{\sum_{k=1}^c \left(\frac{d_{ij}^2}{d_{ik}^2}\right)^{\frac{1}{w-1}}}$$
5. **Partition Entropy (PE):**
   $$\text{PE} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^c u_{ij} \ln(u_{ij}) \quad (\text{untuk } u_{ij} > 0)$$
6. **Fuzzy Partition Coefficient (FPC):**
   $$\text{FPC} = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^c (u_{ij})^2$$
