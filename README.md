# 🥭 Mango Ripeness Detection

Proyek **Computer Vision untuk mendeteksi dan mengklasifikasikan tingkat kematangan buah mangga** (*Unripe*, *Partially Ripe*, *Ripe*) menggunakan model lokal **YOLOv11** (`best.pt`) berbasis arsitektur Object Detection Ultralytics.

Proyek ini telah dikembangkan agar dapat berjalan secara **100% lokal dan offline** tanpa memerlukan API key pihak ketiga untuk inferensi.

---

## 📌 Gambaran Proyek

Proyek ini bertujuan untuk mengimplementasikan model *object detection* dalam mengenali objek buah mangga pada citra digital sekaligus mengidentifikasi tingkat kematangannya.

Model menghasilkan prediksi lokasi objek (*bounding box*), label kelas kematangan, dan nilai keyakinan (*confidence score*).

---

## 🎯 Kelas Deteksi

Model mengenali 3 kelas tingkat kematangan mangga:

| Indeks | Kelas | Deskripsi | Warna Badge |
|:---:|---|---|---|
| 0 | `Partially Ripe` | Mangga setengah matang | Oranye (`#fb923c`) |
| 1 | `Ripe` | Mangga matang | Kuning Emas (`#facc15`) |
| 2 | `Unripe` | Mangga mentah | Hijau (`#4ade80`) |

---

## 🤖 Model & Dataset

| Informasi | Detail |
|---|---|
| **Model Bobot** | `runs/detect/runs/mango/yolo11n_mango/weights/best_v2.pt` |
| **Arsitektur Dasar** | YOLO11 Nano (`yolo11n.pt`) |
| **Tugas** | Object Detection |
| **Framework** | Ultralytics YOLO & Supervision |
| **Dataset Sumber** | Carabao Ripeness Level v19 (582 train, 191 valid, 191 test) |

---

## 🛠️ Teknologi yang Digunakan

* **Python 3.12**
* **Ultralytics YOLO** : Pelatihan model dan inferensi lokal
* **Supervision** : Visualisasi dan annotator bounding box
* **OpenCV & Pillow** : Pemrosesan dan manipulasi citra
* **Streamlit** : Antarmuka dashboard web interaktif

---

## 📂 Struktur Proyek

```text
Mango-Ripeness-Detection/
│
├── .streamlit/
│   └── config.toml           # Konfigurasi batas upload file Streamlit (5MB)
├── datasets/                 # Dataset pelatihan YOLOv11
│   ├── train/ (images, labels)
│   ├── valid/ (images, labels)
│   ├── test/  (images, labels)
│   └── data.yaml             # Konfigurasi kelas dan path dataset
├── images/                   # Gambar sampel untuk pengujian inferensi
├── results/                  # Hasil output visualisasi deteksi bounding box
├── runs/                     # Hasil pelatihan lokal (termasuk weights/best_v2.pt)
├── train.py                  # Skrip pelatihan model YOLO11 (lokal CPU)
├── train_colab.ipynb         # Notebook pelatihan YOLO11 di Google Colab (GPU T4)
├── cek_model.py              # Memeriksa ketersediaan bobot model lokal best_v2.pt
├── uji_model.py              # Menjalankan deteksi batch pada gambar di folder images/
├── app.py                    # Web dashboard interaktif (Streamlit)
├── requirements.txt          # Daftar dependensi Python
└── README.md                 # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan

### 1. (Opsional) Melatih Model Ulang
* **Melalui Google Colab (Direkomendasikan - GPU T4, ~5-10 menit):**  
  Buka file [train_colab.ipynb](train_colab.ipynb) di Google Colab, jalankan semua cell, lalu unduh `best.pt`.
* **Melalui Laptop Lokal (CPU):**
  ```powershell
  python train.py
  ```
*File bobot terbaik akan disimpan otomatis di folder `runs/`.*

### 2. Memeriksa Ketersediaan Model
Untuk memastikan file `best_v2.pt` terbaca dan siap:
```powershell
python cek_model.py
```

### 3. Pengujian Deteksi Batch (Citra Lokal)
Menguji model terhadap sampel citra mangga di dalam folder `images/`:
```powershell
python uji_model.py
```
*Hasil anotasi gambar akan disimpan ke folder `results/`.*

### 4. Menjalankan Dashboard Web (Streamlit)
Membuka aplikasi web interaktif di peramban (browser):
```powershell
streamlit run app.py
```

Fitur dashboard:
* **Deteksi Offline**: Berjalan sepenuhnya menggunakan model lokal `best.pt`.
* **Multi-upload**: Unggah hingga 5 gambar sekaligus (JPG, PNG, WebP hingga 5MB).
* **Slider Threshold**: Atur confidence threshold (0.05 - 1.00) secara fleksibel.
* **Analisis Kematangan**: Ringkasan dominan tingkat kematangan dan kartu detail per buah.
