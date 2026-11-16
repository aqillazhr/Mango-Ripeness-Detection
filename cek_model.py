import os
from pathlib import Path
from ultralytics import YOLO

MODEL_CANDIDATES = [
    Path("runs/detect/runs/mango/yolo11n_mango/weights/best_v2.pt"),
    Path("runs/mango/yolo11n_mango/weights/best_v2.pt"),
    Path("runs/mango/weights/best_v2.pt"),
    Path("best_v2.pt"),
    Path("runs/detect/runs/mango/yolo11n_mango/weights/best.pt"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

print("=" * 60)
print("     PENGECEKAN MODEL LOKAL YOLO11 (best_v2.pt)")
print("=" * 60)

if MODEL_PATH is None or not MODEL_PATH.exists():
    print("\n[ERROR] File model lokal (best_v2.pt) TIDAK DITEMUKAN!")
    print("Lokasi yang dicari:")
    for p in MODEL_CANDIDATES:
        print(f" - {p}")
    print("\nSilakan pastikan file best_v2.pt sudah ada di salah satu lokasi di atas.")
    exit(1)

print(f"\nMenemukan file model di: {MODEL_PATH}")
print(f"Ukuran file: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")
print("Sedang memuat model ke memori...")

try:
    model = YOLO(str(MODEL_PATH))
    print("\n[OK] Model lokal berhasil dimuat dan siap digunakan!")
    print(f"Model Path     : {MODEL_PATH}")
    print(f"Tipe Model     : Object Detection (Ultralytics YOLO)")
    print(f"Arsitektur     : YOLO11 ({model.task})")
    print(f"Jumlah Kelas   : {len(model.names)}")
    print(f"Daftar Kelas   : {model.names}")
    print("\nModel siap digunakan untuk 'uji_model.py' dan 'app.py' tanpa koneksi internet.")
except Exception as e:
    print(f"\n[ERROR] Gagal memuat model: {e}")
