# 🥭 Mango Ripeness Detection

Proyek **Computer Vision untuk mendeteksi dan mengklasifikasikan tingkat kematangan buah mangga** menggunakan model pretrained CARABAO Ripeness Level yang dipublikasikan oleh MangoSort AI melalui Roboflow Universe (carabao-ripeness-level-0x9me/19).

Model ini menggunakan arsitektur YOLOv11 (Object Detection) dengan tiga kelas tingkat kematangan: *Unripe* (mentah), *Partially Ripe* (setengah matang), dan *Ripe* (matang).

## 📌 Gambaran Proyek

Proyek ini bertujuan untuk mengimplementasikan model *object detection* dalam mengenali objek buah mangga sekaligus mengidentifikasi tingkat kematangannya berdasarkan citra digital.

Model menghasilkan prediksi lokasi objek (*bounding box*), label kelas kematangan, dan nilai *confidence score*.

## 🎯 Kelas Deteksi

Model menyediakan tiga kelas tingkat kematangan:

| Class | Description | Warna Badge |
|---|---|---|
| Unripe | Mangga mentah | Hijau (#4ade80) |
| Partially Ripe | Mangga setengah matang | Oranye (#fb923c) |
| Ripe | Mangga matang | Kuning Emas (#facc15) |

## 🤖 Model Pretrained

Model yang digunakan dalam proyek ini:

| Informasi | Detail |
|---|---|
| Nama Model | CARABAO Ripeness Level |
| Model ID | carabao-ripeness-level-0x9me/19 |
| Arsitektur | YOLOv11 |
| Tugas | Object Detection |
| Platform | Roboflow Universe |
| Publisher | MangoSort AI |

**Sumber Model:** [CARABAO Ripeness Level Model > Overview – Roboflow Universe](https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me)

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python 3.12 | Bahasa pemrograman utama |
| Roboflow Inference | Memuat dan menjalankan model pretrained secara lokal |
| Supervision | Menggambar anotasi bounding box dan label pada citra |
| python-dotenv | Membaca konfigurasi API Key dari file .env |
| OpenCV | Membaca, memproses, dan menyimpan gambar |
| Streamlit | Antarmuka dashboard web interaktif |

## 📂 Struktur Proyek

`	ext
Mango-Ripeness-Detection/
│
├── .streamlit/
│   └── config.toml       # Konfigurasi batas upload file Streamlit (5MB)
├── app.py                # Dashboard web interaktif (Streamlit)
├── cek_model.py          # Memeriksa koneksi dan memuat model pretrained
├── uji_model.py          # Menjalankan deteksi batch citra mangga
├── images/               # Gambar input untuk inference
├── results/              # Hasil visualisasi deteksi anotasi
├── requirements.txt      # Daftar dependensi Python
├── README.md             # Dokumentasi proyek
├── .gitignore            # Daftar file yang diabaikan Git
└── test.webp             # Contoh gambar untuk pengujian
`

## ⚙️ Instalasi dan Konfigurasi

### 1. Clone Repository

`ash
git clone https://github.com/aqillazhr/Mango-Ripeness-Detection.git
cd Mango-Ripeness-Detection
`

### 2. Konfigurasi Lingkungan

Pastikan dependensi telah terinstal dan file .env telah disiapkan:

`env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
`

## 🚀 Menjalankan Aplikasi

### 1. Memeriksa Ketersediaan Model

`ash
python cek_model.py
`

### 2. Pengujian Batch Gambar

`ash
python uji_model.py
`

### 3. Menjalankan Dashboard Web Streamlit

`ash
streamlit run app.py
`

Fitur dashboard:
- Sidebar informasi model, jenis tugas, dataset link, daftar kelas, dan threshold keyakinan.
- Unggah hingga 5 berkas citra sekaligus (JPG, JPEG, PNG, WebP hingga 5MB per berkas).
- Visualisasi bounding box proporsional dengan label kematangan.
- Ringkasan status dominan kematangan dan kartu detail per objek mangga.

## 📚 Referensi

- [CARABAO Ripeness Level – Roboflow Universe](https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me)
- [Roboflow Inference Documentation](https://inference.roboflow.com/)
- [Supervision Documentation](https://supervision.roboflow.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
