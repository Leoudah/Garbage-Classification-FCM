# Prompt implementasi alur (salin-tempel ke Antigravity)

Paket rule sudah diekstrak ke root repo. Salin blok di bawah ini utuh, lalu tempel ke Antigravity.

---

```text
/swarm Jalankan dalam mode cepat: buat rencana singkat, bagi kerja ke 3 worker paralel (Worker A, B, C) sesuai skill `.agents/skills/implementasi-alur/SKILL.md` dan tiga rule di `.agents/rules/`, lakukan smoke check ringan di akhir, lalu STOP. Jangan interview panjang, jangan pakai Critic/Auditor, jangan test suite berat, jangan verifikasi browser, dan jangan mengunduh dataset Kaggle penuh di lingkungan lokal.

KONTEKS:
Tahap setup sudah selesai. Kerjakan di repo yang sedang terbuka (fork / clone dari Garbage-Classification-FCM). Acuan kode adalah branch staging commit 367fb93 (24 Nov 2026, 12:25 WITA) yang sudah memuat kerangka full library, tetapi alurnya BELUM siap dijalankan end-to-end. Jangan merancang ulang proyek dan jangan mengulang scaffolding.

Tujuan: selesaikan implementasi alur resmi Dokumen Konteks Sistem — MobileNetV3-Small / (HSV+GLCM) + StandardScaler + PCA + scikit-fuzzy FCM (cmeans & cmeans_predict) + KNN [PCA || alpha * mu] — lalu sambungkan notebook Colab, GUI Streamlit, dan kontrak models/meta.json. Pustaka boleh penuh; rumus internal tetap dijelaskan. Angka hasil eksperimen DILARANG dikarang.

TEMUAN AUDIT STAGING YANG WAJIB DIPERBAIKI (jangan diabaikan):

1) app.py dan notebooks/training.ipynb TIDAK NYAMBUNG ke API src/ yang sebenarnya.
   - fit_pipeline(X, y=None, config=None, feature_extractor=None) dipanggil dengan y_true=, test_size=, random_state= (parameter itu tidak ada) — akan TypeError.
   - Notebook mengimpor compare_scenarios yang tidak ada. app.py mengimpor compare_knn_variants dan compare_feature_extractors yang tidak ada; karena seluruh impor src dibungkus satu try/except, SRC_AVAILABLE bisa menjadi False dan mematikan load_models / fit_pipeline / evaluate_pipeline.
   - Kunci hasil yang benar: results["class_names"], results["data"]["X_train_pca"], results["metrics"]["fcm_train"], results["metrics"]["pca"], results["metrics"]["knn_test"], results["metrics"]["scenarios"], results["fcm"].labels_ / objective_history_. evaluate_pipeline mengembalikan classification_metrics, bukan accuracy di akar.
   - Notebook masih membaca results["X_pca_train"], results.get("fpc"), results["cluster_labels_train"], results.get("classes"). app.py membaca metrics.get("fpc") dan test_eval.get("accuracy"). Samakan semua ke kontrak di .agents/rules/02-skenario-pengujian-evaluasi.md.

2) ANGKA FIKTIF di app.py harus dihapus total.
   - Default/fallback 0.9480, 0.9495, 0.9485, 0.9167.
   - Tabel skenario hardcode 94.80%, 89.20%, 81.50%, 75.30%.
   - Confusion matrix 6x6 hardcode jika data belum ada.
   - Narasi “kesalahan terbanyak pada trash” yang ditulis seolah sudah diukur.
   Jika meta/metrik belum ada, tampilkan keadaan kosong yang jujur: “Hasil eksperimen belum tersedia. Jalankan notebooks/training.ipynb di Google Colab, lalu muat folder models/.”

3) save_models di src/pipeline.py mereset meta_dict di tengah fungsi, sehingga class_names, feature_extractor, alpha, dan scenarios_summary bisa hilang dari meta.json. **kwargs tidak dipakai. Perbaiki agar meta.json mengikuti kontrak rule 02 (dataset 1000/kelas, versi library, fcm_train, knn_test, scenarios 1a/1b/2a/2b, tuning Skenario 3, scenario_4, scenario_5).

4) Skenario belum lengkap.
   - Fungsi run_four_scenarios sudah ada tetapi bisa menelan error (except Exception: pass / hanya menyimpan "error"), mengekstrak ulang MobileNet secara sia-sia, dan belum menyimpan P akhir / n_iter. Notebook memanggil compare_scenarios(data_dir=..., samples_per_class=30) — itu bukan protokol resmi.
   - tune_knn_hyperparameters sudah ada tetapi BELUM dipanggil. Wajib dijalankan sebagai Skenario 3: k in {3,5,7,9}, weights in {uniform,distance}, alpha (usulan {0.5,1.0,1.5,2.0}) via CV pada data LATIH saja. JANGAN tuning di 1.200 data uji. Parameter terbaik dipakai model resmi yang disimpan ke models/.
   - Skenario 4 (opsional, variasi c atau m) belum ada; jika dilewati tulis status “tidak dijalankan” di meta.json, jangan isi angka palsu.
   - Skenario 5: per_class dan confusion_matrix sudah dihitung compute_classification_metrics, tetapi belum disambungkan ke notebook/GUI secara jujur.

5) Protokol dataset resmi belum dipakai di notebook.
   - Training resmi: load_dataset_from_directory(..., samples_per_class=1000, random_state=42) lalu fit_pipeline pada path+label itu (split 80:20 di dalam pipeline). Jangan img_paths[:N], jangan samples_per_class=100 atau 30 untuk tabel resmi.
   - Setup Colab masih dikomentari (git clone, pip install). Aktifkan langkah yang rapi: clone repo kerja, pindah ke root, pip install -r requirements.txt, cek scikit-learn==1.6.1 dan scikit-fuzzy==0.5.0.

6) Contoh manual Kilap/Coklat di notebook memakai U awal yang SALAH (0.85/0.15 dst) dan berhenti setelah 2 iterasi tanpa prediksi KNN. Samakan dengan dokumen kelompok: A=(7,1) plastik, E=(6,2) plastik, B=(9,1) kaca, C=(1,9) kardus, D=(2,7) kertas; U awal A=0.6/0.4, E=0.3/0.7, B=0.5/0.5, C=0.4/0.6, D=0.7/0.3; c=2, w=2, ξ=0.001, MaxIter=100, k=3; prediksi gambar baru X=(6,3).

7) GUI demo: halaman Proses tidak memakai dataset ZIP yang sudah diunggah (selalu generate_synthetic_dataset(15)); terjadi split 80:20 di GUI lalu fit_pipeline mem-split lagi; plot varians bertuliskan Ambang 90% (harusnya pemeriksaan 95%); Info Sistem belum lengkap untuk FAQ dosen, belum menampilkan versi scikit-fuzzy / scikit-learn secara presisi, dan masih menyebut varians 90%. Latih ulang tetap HANYA di st.session_state, jangan menimpa models/.

8) PCA: default resmi n_components=50. variance_threshold=0.95 saat ini hanya disimpan, tidak dipakai. Laporkan explained / cumulative / total variance, dukung n_components float, dan jangan mengabaikan ambang 0.95 secara diam-diam.

Yang sudah bagus di staging (JANGAN dirombak):
- features.py sudah MobileNetV3Small(include_preprocessing=True, input float [0,255], 576-d) plus HSV+GLCM.
- fcm.py sudah wrapper skfuzzy dengan transpose (n_features, n_samples) <-> (n_samples, c), cmeans_predict, FPC, PE (ln), Silhouette.
- config.yaml sudah 1000/kelas, split 0.2, seed 42, c=6, m=2, maxiter=200, error=1e-5, k=5, alpha=1.0, PCA 50.
- requirements.txt sudah pin scikit-learn==1.6.1 dan scikit-fuzzy==0.5.0 (tambahkan pandas, dan seaborn/plotly hanya jika benar-benar dipakai).
- Enam menu Streamlit, validasi c/m/k, peringatan “tidak menimpa models/”, dan cache_resource sudah ada — tinggal disambungkan dan dibersihkan dari angka fiktif.

PEMBAGIAN 3 WORKER:
- Worker A (src/ & config.yaml): perbaiki save_models dan kontrak meta.json; rapikan fit_pipeline / run_four_scenarios / laporan varians PCA; pastikan tune_knn_hyperparameters dan ringkasan skenario 4–5 siap dipanggil; bekukan MobileNet dan (jika mudah) ekstraksi batch; jangan ubah default resmi config.yaml; jangan menelan error secara diam-diam.
- Worker B (app.py): perbaiki impor dan pemanggilan API; hapus semua metrik fiktif; petakan kunci metrik yang benar; sambungkan ZIP ke demo; hindari split ganda; lengkapi 6 halaman, visualisasi, validasi, Info Sistem + FAQ; cache MobileNet; demo hanya di session_state.
- Worker C (notebooks/training.ipynb, requirements.txt, README.md, data/README.md, models/README.md): jadikan notebook runner Colab tipis yang memanggil src/; setup clone+pip+cek versi; sampling 1000/kelas seed 42; contoh manual resmi; Skenario 3 lalu 1a/1b/2a/2b, 4 opsional, 5 per kelas; save_models + verifikasi load_models; rapikan URL clone yang tidak konsisten.

SMOKE CHECK (RINGAN, LALU STOP):
1. python -m py_compile app.py src/features.py src/fcm.py src/pipeline.py
2. Impor publik src.pipeline dan src.fcm berhasil.
3. Uji sintetis kecil (12 sampel / 2 per kelas, atau fitur palsu 26-d jika bobot ImageNet tidak tersedia): scaler → PCA → skfuzzy FCM → [PCA || alpha*mu] → KNN, termasuk cmeans_predict, run_four_scenarios, tune_knn_hyperparameters di latih, serta save_models/load_models ke folder sementara dengan meta.json yang memuat class_names, library_versions, dan kerangka scenarios.
4. Pencarian teks: tidak ada MobileNetV2, 1280, TrashNet, 1.4.2, <username>, dan tidak ada akurasi hardcode 94.80 / 0.9480 / 89.20 / 81.50 / 75.30 / 0.9167 di app.py.
5. app.py tidak mengimpor nama yang tidak ada di src.pipeline.
6. Laporkan file yang diubah, bug yang diperbaiki, perintah menjalankan notebook di Colab, dan hasil smoke check, lalu STOP.
```
