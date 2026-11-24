"""
Skrip Pengunduh Garbage Images Dataset dari Kaggle.
Final Project TLF: Klasifikasi & Clusterisasi Foto Sampah.
Dataset: zlatan599/garbage-dataset-classification (Lisensi MIT)

Penggunaan:
    python data/download_dataset.py [--output-dir data]
"""

import os
import sys
import argparse
import zipfile
import shutil

DATASET_SLUG = "zlatan599/garbage-dataset-classification"

def check_kaggle_installed():
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False

def check_kaggle_credentials():
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    return os.path.exists(kaggle_json)

def download_via_kaggle_api(target_dir):
    print(f"[*] Mengunduh dataset '{DATASET_SLUG}' via Kaggle API...")
    import kaggle
    kaggle.api.authenticate()
    
    os.makedirs(target_dir, exist_ok=True)
    kaggle.api.dataset_download_files(DATASET_SLUG, path=target_dir, unzip=True)
    print(f"[+] Dataset berhasil diunduh dan diekstrak ke: {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Garbage Images Dataset from Kaggle")
    parser.add_argument("--output-dir", default="data", help="Direktori penyimpanan data (default: data)")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.output_dir)
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 70)
    print("  Garbage Images Dataset Downloader - Final Project TLF")
    print("=" * 70)
    print(f"Direktori target: {target_dir}")

    # Cek apakah dataset sudah ada
    possible_paths = [
        os.path.join(target_dir, "Garbage_Dataset_Classification", "images"),
        os.path.join(target_dir, "Garbage Dataset Classification", "images"),
        os.path.join(target_dir, "images")
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) >= 6:
                print(f"[INFO] Dataset terdeteksi sudah tersedia di: {p}")
                print(f"[INFO] Kelas ditemukan: {', '.join(subdirs)}")
                print("[INFO] Tidak perlu mengunduh ulang.")
                return

    if not check_kaggle_installed():
        print("\n[PERINGATAN] Library 'kaggle' belum terpasang.")
        print("Silakan pasang dengan menjalankan: pip install kaggle")
        print("\nAtau unduh manual melalui browser di:")
        print(f"https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"Lalu ekstrak ke folder '{target_dir}/Garbage_Dataset_Classification/images/'.\n")
        return

    if not check_kaggle_credentials():
        print("\n[PERINGATAN] File kredensial 'kaggle.json' tidak ditemukan di ~/.kaggle/")
        print("Langkah mendapatkan kredensial Kaggle:")
        print("1. Buka https://www.kaggle.com -> Masuk ke Akun Anda -> 'Settings'")
        print("2. Scroll ke bagian 'API' -> Klik 'Create New Token' (kaggle.json akan diunduh)")
        print(f"3. Simpan file kaggle.json di direktori: {os.path.join(os.path.expanduser('~'), '.kaggle')}")
        print("\nAtau unduh manual melalui browser di:")
        print(f"https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"Lalu ekstrak ke folder '{target_dir}/Garbage_Dataset_Classification/images/'.\n")
        return

    try:
        download_via_kaggle_api(target_dir)
    except Exception as e:
        print(f"\n[ERROR] Gagal mengunduh dataset: {e}")
        print("\nSilakan unduh manual melalui browser di:")
        print(f"https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"Lalu ekstrak file zip ke folder '{target_dir}/Garbage_Dataset_Classification/images/'.\n")

if __name__ == "__main__":
    main()
