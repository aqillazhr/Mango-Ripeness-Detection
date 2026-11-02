import os
from dotenv import load_dotenv

load_dotenv()

from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 1. Pastikan API Key tersedia
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY belum terbaca di file .env!")

MODEL_ID = "mango-aiyie/1"

# 2. Inisialisasi klien inferensi Roboflow
print("Sedang memeriksa ketersediaan model...")

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key,
)
client.configure(InferenceConfiguration(api_key_transport="header"))

# 3. Uji inferensi ringan untuk memastikan model aktif
candidate_images = ["images/mango-344502_640.jpg", "test.webp"]
if os.path.exists("images"):
    candidate_images += [
        os.path.join("images", f)
        for f in os.listdir("images")
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))
    ]

test_image = next((img for img in candidate_images if os.path.exists(img)), None)

try:
    if test_image:
        res = client.infer(test_image, model_id=MODEL_ID)
        print("\n[OK] Model berhasil dimuat dan siap digunakan!")
        print(f"Model ID       : {MODEL_ID}")
        print("Jenis Tugas    : Classification (Klasifikasi Citra)")
        print(f"Kelas Prediksi : {', '.join(res.get('predictions', {}).keys())}")
        print(f"Sampel Gambar  : {test_image}")
        print(f"Hasil Uji Coba : {res.get('predicted_classes', ['-'])[0]}")
    else:
        print("\n[OK] Klien model berhasil diinisialisasi untuk MODEL_ID:", MODEL_ID)
        print("Peringatan: Tidak ditemukan berkas gambar contoh untuk inferensi uji coba.")
except Exception as e:
    print(f"\n[ERROR] Gagal mengakses model {MODEL_ID}: {e}")
