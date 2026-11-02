import os
import cv2
import supervision as sv
from inference import get_model

# 1. Konfigurasi
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "carabao-ripeness-level-0x9me/19"
IMAGE_PATH = "test.webp"
OUTPUT_PATH = "hasil_deteksi.webp"

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY belum terbaca!")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Foto tidak ditemukan: {IMAGE_PATH}"
    )

# 2. Load model CARABAO
print("Memuat model CARABAO...")

model = get_model(
    model_id=MODEL_ID,
    api_key=API_KEY
)

print("Model berhasil dimuat!")

# 3. Jalankan deteksi pada foto
print("\nMemulai deteksi mangga...")

results = model.infer(IMAGE_PATH)
predictions = results[0]

# 4. Baca gambar
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Gambar gagal dibaca oleh OpenCV.")

# 5. Konversi hasil deteksi
detections = sv.Detections.from_inference(predictions)

# 6. Buat label kelas dan confidence
labels = [
    f"{pred.class_name} {pred.confidence:.2f}"
    for pred in predictions.predictions
]

# 7. Gambar bounding box
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

# 8. Simpan gambar hasil deteksi
cv2.imwrite(OUTPUT_PATH, annotated_image)

# 9. Tampilkan hasil di Terminal
print("\n===== HASIL DETEKSI =====")

if predictions.predictions:
    for pred in predictions.predictions:
        print(
            f"Kelas: {pred.class_name} | "
            f"Confidence: {pred.confidence:.2%}"
        )
else:
    print("Tidak ada objek mangga yang terdeteksi.")

print(f"\nHasil gambar disimpan di: {OUTPUT_PATH}")
