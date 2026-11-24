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

## 4. Penggunaan pada Training & Evaluasi

- **Framing Masalah:** Model utama **Fuzzy C-Means (FCM)** bekerja secara **unsupervised** pada fitur PCA tanpa membaca label nama folder saat clustering.
- Sifat keanggotaan fuzzy $\mu_{ij}$ menangani ambiguitas material sampah (misalnya kemiripan visual kardus kotor dengan kertas, atau plastik transparan dengan kaca).
- Label kelas asli dari subfolder hanya digunakan untuk:
  1. Melatih classifier **KNN** pemetaan dengan input gabungan `[fitur_pca, mu * mu_weight]`.
  2. Menghitung metrik evaluasi eksternal (Akurasi, Precision macro, Recall macro, F1 macro, dan Confusion Matrix).
- Untuk training ringan dan representatif, dataset dibagi dengan **stratified train/test split (80:20)** dengan subset seimbang ~100 sampel per kelas.
