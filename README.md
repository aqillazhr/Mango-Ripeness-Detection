# 🥭 Mango Detection & Classification

Proyek **Computer Vision untuk mengklasifikasikan kondisi dan kesegaran buah mangga (Fresh Mango vs Rotten Mango)** menggunakan model pretrained yang dipublikasikan oleh parekh melalui Roboflow Universe (mango-aiyie).

Model ini menggunakan arsitektur Vision Transformer (ViT) Image Classification dengan dua kelas: *Fresh Mango* (mangga segar) dan *Rotten Mango* (mangga busuk).

Proyek ini berfokus pada implementasi dan pengujian model pretrained tersebut menggunakan Roboflow Inference SDK (inference-sdk).

## 📌 Gambaran Proyek

Proyek ini bertujuan untuk mengimplementasikan model *image classification* dalam mengidentifikasi kondisi kelayakan dan kesegaran mangga berdasarkan citra digital.

Model menghasilkan probabilitas kelas kondisi buah serta nilai *confidence score*.

**Catatan:** Model yang digunakan merupakan model pretrained dari pihak ketiga (parekh/mango-aiyie). Proyek ini berfokus pada integrasi, pengujian, dan implementasi antarmuka dashboard berbasis Streamlit.

## 🎯 Kelas Deteksi

Model menyediakan dua kelas klasifikasi kondisi:

| Class | Description | Status Kualitas |
|---|---|---|
| Fresh Mango | Mangga segar | Prima, layak dikonsumsi/dipasarkan |
| Rotten Mango | Mangga busuk/rusak | Rusak, perlu disortir |

## 🤖 Model Pretrained

Model yang digunakan dalam proyek ini:

| Informasi | Detail |
|---|---|
| Nama Model | mango |
| Model ID | mango-aiyie/1 |
| Arsitektur | Vision Transformer (ViT) |
| Tugas | Image Classification |
| Platform | Roboflow Universe |
| Publisher | parekh |

**Sumber Model:** [mango Model > Overview – Roboflow Universe](https://universe.roboflow.com/parekh/mango-aiyie)

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python 3.12 | Bahasa pemrograman utama |
| Roboflow Inference SDK | Mengakses model pretrained via Roboflow Inference |
| python-dotenv | Membaca konfigurasi dari file .env |
| OpenCV | Membaca, memproses, dan memvisualisasikan anotasi gambar |
| Streamlit | Antarmuka dashboard web interaktif |

## 📂 Struktur Proyek

`	ext
Mango-Ripeness-Detection/
│
├── .streamlit/
│   └── config.toml       # Konfigurasi batas upload file Streamlit (5MB)
├── app.py                # Dashboard web interaktif (Streamlit)
├── cek_model.py          # Memeriksa koneksi dan ketersediaan model pretrained
├── uji_model.py          # Menjalankan klasifikasi batch citra mangga
├── images/               # Gambar input untuk inference
├── results/              # Hasil visualisasi klasifikasi
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

### 2. Membuat dan Mengaktifkan Virtual Environment

**Windows (PowerShell):**

`powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
`

**macOS / Linux:**

`ash
python3.12 -m venv .venv
source .venv/bin/activate
`

### 3. Menginstal Dependensi

`ash
pip install -r requirements.txt
`

### 4. Konfigurasi API Key Roboflow

Buat file .env di root repository:

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

Script akan:
1. Membaca gambar dari folder images/.
2. Menjalankan klasifikasi kondisi mangga (Fresh Mango vs Rotten Mango).
3. Menambahkan visual banner hasil klasifikasi pada citra.
4. Menyimpan hasil anotasi ke direktori 
esults/.

### 3. Menjalankan Dashboard Web Streamlit

`ash
streamlit run app.py
`

Fitur dashboard:
- Unggah hingga 5 berkas citra sekaligus (JPG, JPEG, PNG, WebP hingga 5MB per berkas).
- Visualisasi citra dengan banner status klasifikasi.
- Kartu probabilitas detail per kelas (Fresh Mango vs Rotten Mango) beserta indikator progress bar.
- Rekomendasi kondisi dan kelayakan konsumsi buah.

## 📚 Referensi

- [mango Model > Overview – Roboflow Universe](https://universe.roboflow.com/parekh/mango-aiyie)
- [Roboflow Inference SDK Documentation](https://inference.roboflow.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 Atribusi Model

Model klasifikasi mangga yang digunakan dalam proyek ini dipublikasikan oleh **parekh** melalui Roboflow Universe (mango-aiyie). Seluruh hak cipta, lisensi, dan ketentuan model mengikuti informasi dari publisher pada halaman Roboflow aslinya.
