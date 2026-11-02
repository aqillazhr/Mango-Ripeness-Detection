import os
from dotenv import load_dotenv

load_dotenv()

from inference import get_model

# 1. Pastikan API Key tersedia
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY belum terbaca di file .env!")

MODEL_ID = "carabao-ripeness-level-0x9me/19"

# 2. Ambil model CARABAO dari Roboflow
print("Sedang memuat model CARABAO...")

try:
    model = get_model(model_id=MODEL_ID, api_key=api_key)
    print("\n[OK] Model berhasil dimuat dan siap digunakan!")
    print(f"Model ID       : {MODEL_ID}")
    print("Tipe Model     : Object Detection")
    print("Arsitektur     : YOLOv11")
    print("Kelas Prediksi : Unripe, Partially Ripe, Ripe")
    print("Sumber Dataset : https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me")
except Exception as e:
    print(f"\n[ERROR] Gagal memuat model: {e}")
