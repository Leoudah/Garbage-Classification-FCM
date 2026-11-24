# Direktori Data — Garbage Images Dataset

Direktori ini digunakan untuk menyimpan dataset citra sampah **Garbage Images Dataset** dari Kaggle.

> **PENTING:** File citra mentah berukuran besar tidak boleh di-commit ke Git repository (sudah dikecualikan melalui `.gitignore`).

---

## 1. Sumber Dataset Acuan: Garbage Images Dataset

- **Tautan Kaggle:** [Garbage Dataset Classification](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification)
- **Slug Dataset:** `zlatan599/garbage-dataset-classification`
- **Lisensi:** MIT License
- **Kategori / Kelas:**
  1. `cardboard` (kardus)
  2. `glass` (kaca)
  3. `metal` (logam)
  4. `paper` (kertas)
  5. `plastic` (plastik)
  6. `trash` (sampah residu/lainnya)

---

## 2. Struktur Folder yang Diharapkan

Setelah diunduh dan diekstrak, letakkan dataset di folder `data/Garbage_Dataset_Classification/images/` dengan struktur subfolder per kelas dan file metadata:

```
data/
├── README.md
├── download_dataset.py
└── Garbage_Dataset_Classification/
    ├── metadata.csv
    └── images/
        ├── cardboard/
        │   ├── cardboard_001.jpg
        │   └── ...
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

---

## 3. Cara Mengunduh Dataset

### Cara A: Menggunakan Skrip Otomatis (`download_dataset.py`)
Pastikan Anda telah memiliki API token Kaggle (`kaggle.json` yang diletakkan di `~/.kaggle/kaggle.json` atau `%USERPROFILE%\.kaggle\kaggle.json`).

Jalankan perintah berikut di terminal:
```bash
python data/download_dataset.py
```

Skrip akan secara otomatis mengunduh dataset dari Kaggle dan mengekstraknya ke direktori `data/Garbage_Dataset_Classification/`.

### Cara B: Pengunduhan Manual via Browser
1. Buka tautan Kaggle: [Garbage Images Dataset](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification).
2. Unduh file zip dataset.
3. Ekstrak isi file zip tersebut ke dalam folder `data/Garbage_Dataset_Classification/`.

---

## 4. Protokol Pengambilan Sampel & Evaluasi

- **Sampling Representatif:** Random sampling 1.000 citra per kelas (total 6.000 citra untuk 6 kelas) dengan `random_state=42`.
- **Anti Data Leakage:** Stratified split 80:20 (4.800 data latih, 1.200 data uji) dilakukan di awal sebelum ekstraksi fitur dan fitting model.
- **Ekstraksi Fitur:** Fitur diekstraksi menggunakan MobileNetV3-Small (576 dimensi) dengan jalur pembanding Handcrafted (HSV + GLCM).
- **Clusterisasi & Klasifikasi:** FCM (`scikit-fuzzy`) dilatih tanpa label pada ruang PCA data latih. Label subfolder hanya digunakan untuk melatih KNN dan menghitung metrik evaluasi klasifikasi (Akurasi, Presisi, Recall, F1, Confusion Matrix).
