---
trigger: glob
globs: "app.py, models/**"
description: "Checklist 6 halaman Streamlit, validasi masukan, visualisasi, keamanan demo, dan kesiapan tanya jawab dosen."
---

# GUI Streamlit dan demonstrasi (tahap implementasi)

Aplikasi wajib dapat dipakai tanpa menyunting kode. Enam halaman tetap berada di `app.py` (mode single-file di `staging` sudah diterima; jangan memaksa pecah ke `pages/` kecuali ada alasan kuat). Seluruh teks UI berbahasa Indonesia. Logika prediksi, pelatihan, dan evaluasi **diimpor dari `src/`**, bukan disalin ulang di `app.py`.

## Impor dan ketersediaan `src/` (bug kritis di `staging`)

Blok impor di `app.py` saat ini merujuk `compare_knn_variants` dan `compare_feature_extractors` yang **tidak ada** di `src/pipeline.py`. Karena seluruh impor dibungkus satu `try/except ImportError`, kegagalan itu membuat `SRC_AVAILABLE = False` dan mematikan akses ke `fit_pipeline`, `load_models`, dan `evaluate_pipeline`.

Perbaiki:

- Impor hanya nama yang benar-benar ada (`fit_pipeline`, `evaluate_pipeline`, `predict` / `pipeline_predict`, `load_models`, `load_default_config`, `run_four_scenarios`, `tune_knn_hyperparameters` jika diperlukan).
- Jangan biarkan satu nama usang menonaktifkan seluruh paket `src`.
- Panggil `fit_pipeline(X, y=..., config=...)` — bukan `y_true=`.
- Baca metrik sesuai kontrak di rule 02. Jangan memakai default numerik fiktif.

Cache MobileNetV3-Small dengan `@st.cache_resource`. Lebih baik memanggil `src.features.get_mobilenet_model()` (atau berbagi cache yang sama) daripada merawat salinan konstruktor yang bisa menyimpang.

## Enam halaman (sesuai Bagian 8 Dokumen Konteks)

Label menu boleh memakai ikon, tetapi nama konsepnya tetap: **Beranda**, **Input Data**, **Proses**, **Hasil**, **Evaluasi**, **Info Sistem**.

### 1. Beranda

- Nama sistem, deskripsi singkat, tujuan, dan navigasi ke halaman lain.
- Framing masalah pemilahan sampah ambigu di TPS / Bank Sampah.
- Ringkasan enam kelas: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- Satu paragraf alur: citra → MobileNetV3-Small / (HSV+GLCM) → StandardScaler → PCA → FCM (`μ`) → KNN `[PCA || α·μ]`.
- Status model yang jujur: disk `models/`, sesi interaktif, atau belum tersedia. Jangan menampilkan akurasi jika belum ada hasil.

### 2. Input Data

- Unggah satu (atau beberapa) gambar uji: **JPG / JPEG / PNG**, batas ukuran (usulan 10 MB) sudah ada dan boleh dipertahankan.
- Pesan galat jelas jika berkas kosong, format tidak didukung, atau citra rusak / tidak dapat dibaca.
- Preview thumbnail.
- Tombol sampel bawaan boleh ada, tetapi dilabeli sebagai contoh sintetis / ilustrasi, bukan data Kaggle.
- Unggah ZIP / folder dataset untuk pelatihan atau evaluasi ulang: subfolder kelas `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- **Celah `staging`:** ZIP tersimpan di `st.session_state["dataset_samples"]` tetapi halaman Proses **tidak memakainya** dan selalu memanggil `generate_synthetic_dataset(15, ...)`. Jika pengguna sudah mengunggah dataset, latih demo dari data itu (subset wajar, misalnya puluhan hingga ratusan citra). Jika belum ada unggahan, barulah fallback sintetis, dengan label yang jujur bahwa itu demo bukan eksperimen resmi.
- Penjelasan singkat arti kelas, fitur, serta parameter `c`, `m`, `ξ`, `k`, dan `α`.

### 3. Proses

Kontrol parameter: `c`, `m`, `ξ` (`error`), `k`, `α`, plus `weights` dan `n_components` (sudah ada — pertahankan). Nilai awal mengikuti tabel resmi (`c=6`, `m=2.0`, `ξ=1e-5`, `k=5`, `α=1.0`). Slider PCA demo boleh lebih kecil dari 50 demi kecepatan, tetapi harus bertuliskan bahwa itu **mode demo**, bukan protokol resmi 50 komponen / 6.000 citra.

Validasi dan pesan galat:

- `c < 2`
- `m <= 1`
- `k < 1`
- data belum diunggah / belum tersedia (jika mode unduhan dataset dipilih)

Tombol “Jalankan” menampilkan progres (ekstraksi → split → scaler/PCA → FCM / kurva `P` → KNN).

**Keamanan demo (wajib):** hasil latih ulang hanya masuk `st.session_state` (misalnya `session_model`, `training_results`). **Jangan** memanggil `save_models()` dari tombol ini dan **jangan** menimpa folder `models/`.

**Celah `staging` yang harus dirapikan:**

- `fit_pipeline(X_tr, y_true=y_tr, ...)` akan error karena parameternya `y`.
- Data sudah di-split 80:20 di GUI, lalu `fit_pipeline` mem-split lagi. Lebih bersih: serahkan split ke `fit_pipeline`, atau kirim fitur yang sudah di-split dengan `test_size` yang konsisten tanpa pecah dua kali. Evaluasi uji demo harus memakai hold-out yang sama.
- Pesan sukses memakai `test_eval.get('accuracy', 0.9167)` dan `metrics.get('fpc', 0)` padahal `fpc` bersarang di `metrics["fcm_train"]`. Jika pelatihan gagal atau kunci absen, tampilkan error / “metrik belum dihitung”, bukan 91,67%.

### 4. Hasil

Untuk gambar yang diunggah (memakai model disk atau model sesi, dengan prioritas yang sudah ada):

- Kelas prediksi KNN.
- Tabel derajat keanggotaan `μ` terhadap setiap cluster, plus bar chart jika membantu.
- Scatter PC1–PC2: sebaran data latih (opsional mewarnai cluster `argmax μ` atau kelas), pusat `V`, dan posisi citra baru.
- Grafik fungsi objektif `P` versus iterasi.
- Interpretasi singkat: kelas prediksi + cluster dengan `μ` terbesar, tanpa mengaku cluster itu “nama kelas”.

Jika model belum ada, tampilkan pesan ramah (arahkan ke Proses atau notebook Colab), bukan traceback.

### 5. Evaluasi

- Metrik FCM: `P` akhir, iterasi, FPC, PE, Silhouette — diambil dari `metrics["fcm_train"]` atau `meta.json["fcm_train"]`.
- Metrik klasifikasi uji: akurasi, presisi, recall, F1 (makro / berbobot dan per kelas) dari `metrics["knn_test"]` atau `evaluate_pipeline(...)["classification_metrics"]`.
- Confusion matrix dari data nyata, dengan label kelas yang benar.
- Tabel perbandingan 4 skenario 1a / 1b / 2a / 2b dari `metrics["scenarios"]` atau `meta.json["scenarios"]` (bukan kunci `"comparisons"` yang belum ditulis).
- Jika `meta.json` belum ada atau skenario gagal: keadaan kosong yang jujur, **tanpa** tabel 94,80% / 89,20% / 81,50% / 75,30% dan tanpa matrix hardcode.
- Analisis `trash` hanya jika confusion matrix nyata sudah ada (Skenario 5). Jika belum, jangan mengunci kesimpulan.
- Ringkasan Skenario 3 (parameter terbaik hasil CV latih) boleh ditampilkan dari `meta.json["tuning"]`.

### 6. Info Sistem

Wajib siap untuk tanya jawab dosen (Bagian 12 Dokumen Konteks), dengan satu penyesuaian resmi: FCM memakai **`scikit-fuzzy`**, bukan “dihitung sendiri dengan NumPy sebagai implementasi utama”.

Isi minimum:

- Metode FCM dan KNN, cara integrasi `[PCA || α·μ]`, variabel, dan parameter default.
- Rumus `V`, `d²`, `P`, `μ`, `μ` sampel baru, Euclidean KNN, FPC, PE (`ln`), Silhouette.
- Penjelasan full library: TensorFlow/Keras, scikit-learn, scikit-fuzzy, OpenCV / scikit-image, Streamlit, Pillow, Matplotlib (plus pandas/seaborn/plotly hanya jika terpasang dan dipakai).
- Kontrak transpose scikit-fuzzy `(n_features, n_samples)` ⟷ `(n_samples, c)`.
- Kejujuran kriteria berhenti pustaka (perubahan `U`) versus usulan dokumen (`|ΔP| < ξ`).
- Contoh mini Kilap/Coklat (data, `U` awal resmi, X baru = (6, 3), `k=3`) — ringkas, angka inisialisasi harus sama dengan dokumen.
- Petunjuk langkah: Colab → unduh `models/` → `streamlit run app.py`.
- Tabel versi lingkungan: sertakan **scikit-fuzzy** dan versi `scikit-learn` yang terdeteksi (target `1.6.1`), bukan teks “Tersedia” saja. Ganti sisa “varians 90%” menjadi pemeriksaan 95%.

### FAQ dosen (siapkan jawaban singkat di Info Sistem)

1. **Dari mana `μ` gambar baru berasal? Apakah FCM dijalankan lagi?** Tidak. FCM hanya dilatih pada data latih, pusat `V` disimpan. Untuk gambar baru, `μ` dihitung dari jarak ke `V` via `cmeans_predict` / rumus pembaruan `U`. Iterasi pelatihan tidak diulang.
2. **Mengapa ini integrasi, bukan dua metode terpisah?** Karena masukan KNN adalah PCA yang digabung `α·μ`. Jika modul FCM dihilangkan, vektor masukan berubah.
3. **Mengapa data dibagi dulu?** Agar tidak terjadi kebocoran. Scaler, PCA, dan pusat cluster dipelajari dari data latih saja.
4. **Mengapa piksel MobileNetV3-Small tetap `[0, 255]`?** Karena `include_preprocessing=True` (lapisan Rescaling bawaan). Model mengharapkan float `[0, 255]`, bukan `[-1, 1]` di luar model.
5. **Mengapa memakai pustaka FCM, bukan ditulis manual?** Dosen mengizinkan pustaka asal prosesnya dipahami. Kelompok memakai `scikit-fuzzy`, tetapi tetap menelusuri rumus, iterasi, fungsi objektif, FPC, PE, dan contoh manual saat demonstrasi.
6. **Mana yang tak terawasi, mana yang terawasi?** FCM tak terawasi (tanpa label). KNN terawasi (label asli). Keduanya diikat oleh `μ` sebagai fitur tambahan.
7. **Apakah satu cluster sama dengan satu kelas?** Tidak otomatis. `c=6` hanya diselaraskan; nomor cluster bisa tertukar.
8. **Mengapa perlu `α`?** `μ` berada di `[0, 1]`, sedangkan PCA bisa berskala dan berdimensi lebih besar. Tanpa `α`, kontribusi `μ` bisa terlalu lemah. Nilainya dituning di data latih.
9. **Mengapa `k` tidak langsung ditetapkan?** `k` bergantung pada data, jadi 3, 5, 7, dan 9 dicoba lewat validasi pada data latih.
10. **Mengapa dua jalur fitur?** Skenario 2 membandingkan fitur deep vs handcrafted, masing-masing dengan dan tanpa FCM, pada split yang sama.

## Prinsip tampilan dan keamanan

- Antarmuka rapi, konsisten, informatif.
- Setiap masukan tidak valid memunculkan pesan kesalahan yang jelas, bukan traceback mentah ke pengguna.
- Jangan menimpa `models/` dari GUI.
- Jangan menampilkan metrik rekaan.
- Visualisasi resmi: tabel `μ`, scatter PC1–PC2 + pusat, grafik `P` vs iterasi, confusion matrix, tabel 4 skenario. Matplotlib sudah cukup; tambahkan seaborn/plotly hanya jika benar-benar dipakai dan dicantumkan di `requirements.txt`.
