# Cara memakai paket rule tahap implementasi

Paket ini berisi konteks, rules, skill, dan prompt Antigravity untuk **tahap implementasi alur** Final Project TLF Kelompok 5. Setup sudah selesai; jangan memakai paket starter lama yang masih menyebut MobileNetV2 atau FCM manual NumPy sebagai implementasi utama.

## 1. Siapkan repo kerja

Checkout kode terbaru teman (bukan scaffolding lama):

```bash
# Pilihan A — repo utama kelompok (disarankan)
git clone https://github.com/Leoudah/Garbage-Classification-FCM.git
cd Garbage-Classification-FCM
git checkout staging
git pull

# Pilihan B — jika Anda bekerja di fork sendiri setelah sync
# git clone https://github.com/YasMbul/Garbage-Classification-FCM.git
# cd Garbage-Classification-FCM
# git checkout staging    # atau dev/deas, jika staging belum di-sync
```

Pastikan Anda berada di commit yang memuat kerangka full library (MobileNetV3-Small + `scikit-fuzzy`). Acuan audit paket ini adalah `staging` commit `367fb93` (24 Nov 2026, 12:25 WITA).

## 2. Ekstrak zip ini ke root repo

File di dalam zip sudah berada di root arsip (bukan di folder pembungkus tambahan). Ekstrak tepat di folder proyek, berdampingan dengan `app.py`, `src/`, dan `config.yaml`:

```bash
unzip antigravity-rules-implementasi-tlf.zip -d .
```

Yang akan muncul:

```text
AGENTS.md
CARA-PAKAI.md
PROMPT-IMPLEMENTASI.md
.agents/rules/01-konteks-arsitektur-pipeline.md
.agents/rules/02-skenario-pengujian-evaluasi.md
.agents/rules/03-gui-streamlit-demo.md
.agents/skills/implementasi-alur/SKILL.md
```

Jika diminta menimpa `AGENTS.md` lama, izinkan: isi paket ini untuk tahap implementasi, bukan tahap setup.

## 3. Tempel prompt ke Antigravity

1. Buka proyek itu di Antigravity.
2. Buka `PROMPT-IMPLEMENTASI.md`.
3. Salin seluruh blok yang dimulai dengan `/swarm` (di dalam pagar kode).
4. Tempel dan jalankan.

Antigravity akan memakai mode cepat: rencana singkat, 3 worker paralel, smoke check ringan, lalu berhenti. Ia **tidak** men-training 6.000 citra di lokal dan **tidak** boleh mengarang angka akurasi.

Setelah selesai, baca ringkasan file yang diubah dan hasil smoke check. Commit perubahan ke branch kerja Anda (`staging` atau `dev/deas`).

## 4. Jalankan training resmi di Google Colab

Eksperimen penuh (1.000 citra/kelas, 4 skenario, tuning, artefak `models/`) dilakukan di Colab, bukan di laptop saat sesi Antigravity.

1. Buka `notebooks/training.ipynb` di Google Colab.
2. Clone repo **yang sudah berisi hasil implementasi** (bukan commit lama sebelum prompt ini dijalankan).
3. Jalankan sel berurutan: pasang dependensi, unduh dataset Kaggle, contoh manual Kilap/Coklat, training, Skenario 3 lalu 1a/1b/2a/2b, Skenario 4 (opsional), Skenario 5, simpan `models/`.
4. Unduh folder `models/` (`scaler.pkl`, `pca.pkl`, `fcm_centers.npy`, `knn.pkl`, `meta.json`) ke laptop.
5. Di lokal: `pip install -r requirements.txt` lalu `streamlit run app.py`. Pastikan versi `scikit-learn` lokal sama dengan Colab (`1.6.1`).

Angka di halaman Evaluasi, BAB IV, dan Dokumen Konteks baru boleh diisi dari `meta.json` hasil langkah ini.

## 5. (Opsional) Bersihkan file Antigravity sebelum pengumpulan

File rule berguna saat pengembangan, tetapi tidak wajib masuk arsip tugas. Sebelum pengumpulan resmi, Anda boleh menghapus:

- `.agents/`
- `AGENTS.md`
- `PROMPT-IMPLEMENTASI.md`
- `CARA-PAKAI.md`

Yang tetap dikumpulkan: kode aplikasi (`src/`, `app.py`, `config.yaml`, `requirements.txt`, `README.md`), folder `models/` hasil Colab, notebook eksperimen, plus laporan BAB I–V dan bahan presentasi sesuai ketentuan dosen.

## Isi paket (ringkas)

| Berkas | Kegunaan |
| --- | --- |
| `AGENTS.md` | Instruksi workspace tahap implementasi, mode `/swarm`, peta repo, guardrail |
| `.agents/rules/01-konteks-arsitektur-pipeline.md` | Dataset, dua jalur fitur, pipeline full library, rumus, parameter, contoh manual |
| `.agents/rules/02-skenario-pengujian-evaluasi.md` | Skenario 1–5, kontrak `meta.json`, larangan angka fiktif |
| `.agents/rules/03-gui-streamlit-demo.md` | Checklist 6 halaman Streamlit, validasi, demo aman, FAQ dosen |
| `.agents/skills/implementasi-alur/SKILL.md` | Pembagian 3 worker dan smoke check berdasarkan audit `staging` |
| `PROMPT-IMPLEMENTASI.md` | Prompt `/swarm` siap tempel |
| `CARA-PAKAI.md` | Panduan ini |
