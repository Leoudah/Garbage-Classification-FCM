# Cara pakai paket starter Antigravity

Paket ini berisi instruksi untuk **Google Antigravity**, supaya agent bisa scaffolding repo Final Project TLF (foto sampah, PCA + FCM + KNN, GUI Streamlit). Ini bukan source code aplikasinya — cuma brief + aturan kerja.

## Langkah

1. Buat folder proyek kosong, misalnya `fuzzy-sampah`.
2. Extract zip ini, lalu **salin semua isinya ke root folder proyek** (file `AGENTS.md` harus ada di root, bukan di dalam subfolder `antigravity-starter/`).
3. Buka folder itu sebagai workspace di Antigravity.
4. Buka `PROMPT-PERTAMA.md`, salin seluruh prompt di dalamnya, paste ke chat agent, lalu kirim.
5. Setelah agent selesai (smoke check + ringkasan), cek struktur repo dan coba `streamlit run app.py`.

## Mode teamwork: pakai yang cepat

Antigravity punya dua mode tim yang sering ketuker namanya:

| Perintah | Kapan dipakai | Di paket ini |
| --- | --- | --- |
| `/swarm` | Kerja paralel yang terbagi rapi per file, mulai langsung, verifier ringan | **Ini yang dipakai** |
| `/teamwork` | Proyek panjang (jam–hari): interview dulu, milestone, Critic / Challenger / Auditor / Success Auditor | **Jangan** untuk scaffolding ini |

Prompt pertama sudah diawali `/swarm`. Kalau agent malah mulai interview panjang atau pindah ke `~/teamwork_projects/`, hentikan dan kirim ulang prompt dari `PROMPT-PERTAMA.md`.

## Setelah repo jadi

- Biarkan `AGENTS.md`, `SPESIFIKASI-PROYEK.md`, dan folder `.agents/` di repo — masih berguna waktu lanjut fitur.
- `CARA-PAKAI.md` dan `PROMPT-PERTAMA.md` boleh dihapus kalau sudah tidak diperlukan.
- Training dataset penuh, prediksi foto nyata, dan laporan BAB I–V dikerjakan belakangan (biasanya di Google Colab + tulis laporan), bukan di langkah setup ini.
