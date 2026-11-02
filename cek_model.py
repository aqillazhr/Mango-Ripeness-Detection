import os
from dotenv import load_dotenv

load_dotenv()

from inference import get_model

# 1. Pastikan API Key tersedia
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY belum terbaca!")

# 2. Ambil model CARABAO dari Roboflow
print("Sedang mengambil model CARABAO...")

model = get_model(
    model_id="carabao-ripeness-level-0x9me/19",
    api_key=api_key
)

# 3. Konfirmasi model berhasil dimuat
print("\nModel berhasil dimuat!")
print("Tipe model:", type(model).__name__)
