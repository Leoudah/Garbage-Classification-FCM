# Prompt pertama (siap salin)

Salin semua teks di dalam blok di bawah ini, lalu paste ke chat Antigravity di folder proyek (setelah file starter ini ada di root workspace).

```
/swarm Setup scaffolding repo Final Project TLF sesuai file di workspace ini. Ini cuma setup project — pakai mode teamwork CEPAT.

Baca dulu AGENTS.md, SPESIFIKASI-PROYEK.md, lalu ikuti skill `.agents/skills/setup-scaffold/SKILL.md` dan rule `.agents/rules/algoritma-pipeline.md`.

Rencana cukup singkat, lalu pecah kerja ke worker paralel yang file-nya tidak tumpang tindih:
- Worker A — src/: `src/features.py`, `src/fcm.py` (FCM ditulis manual pakai NumPy), `src/pipeline.py`, `config.yaml`
- Worker B — GUI: `app.py` dan halaman Streamlit (Beranda, Upload, Proses, Hasil, Evaluasi, Petunjuk)
- Worker C — docs & wiring: `notebooks/training.ipynb` (siap Colab, import dari `src/`), `README.md`, `requirements.txt` (pin scikit-learn + catatan samakan versi dengan Colab), `.gitignore`, skrip unduh data, placeholder `models/`

Kerjakan di folder workspace ini. Jangan pakai /teamwork default (jangan interview Phase 1, jangan Critic/Challenger/Auditor/Success Auditor, jangan pindah ke ~/teamwork_projects/). Jangan download TrashNet penuh. Jangan bikin test suite berat, jangan verifikasi lewat browser.

Smoke check saja: import dari `src/` berhasil, pipeline jalan di data sintetis kecil, `streamlit run app.py` berhasil start. Setelah itu STOP. Jangan lanjut training penuh, laporan BAB I–V, atau polish panjang. Tutup dengan ringkasan file yang dibuat, cara menjalankan, dan hasil smoke check.
```
