# 🥭 Mango Ripeness Detection

Proyek **Computer Vision untuk mendeteksi dan mengklasifikasikan tingkat kematangan buah mangga** menggunakan model pretrained CARABAO Ripeness Level yang dipublikasikan oleh MangoSort AI melalui Roboflow Universe.

Model ini menggunakan arsitektur YOLOv11 dan memiliki tiga kelas deteksi, yaitu *Unripe* (mentah), *Partially Ripe* (setengah matang), dan *Ripe* (matang).

Proyek ini berfokus pada implementasi dan pengujian model pretrained tersebut menggunakan Roboflow Inference. Proses inferensi gambar dijalankan secara lokal di komputer pengguna (*local inference*).

## 📌 Gambaran Proyek

Proyek ini bertujuan untuk mengimplementasikan model *object detection* dalam mengenali objek mangga sekaligus mengidentifikasi tingkat kematangannya berdasarkan citra digital.

Model menghasilkan prediksi berupa lokasi objek (*bounding box*), label kelas kematangan, dan nilai *confidence score*.

**Catatan:** Model yang digunakan merupakan model pretrained dari pihak ketiga. Proyek ini tidak mengklaim pengembangan maupun pelatihan model CARABAO dari awal. Fokus proyek adalah implementasi dan pengujian model yang telah tersedia.

## 🎯 Kelas Deteksi

Model menyediakan tiga kelas tingkat kematangan:

| Class | Description |
|---|---|
| `Unripe` | Mangga mentah |
| `Partially Ripe` | Mangga setengah matang |
| `Ripe` | Mangga matang |

## 🤖 Model Pretrained

Model yang digunakan dalam proyek ini:

| Informasi | Detail |
|---|---|
| Nama Model | CARABAO Ripeness Level |
| Model ID | `carabao-ripeness-level-0x9me/19` |
| Arsitektur | YOLOv11 |
| Tugas | Object Detection |
| Platform | Roboflow Universe |
| Publisher | MangoSort AI |

**Sumber Model:** [CARABAO Ripeness Level – Roboflow Universe](https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me)

Model diperoleh dari Roboflow Universe dan dimuat menggunakan Roboflow Inference. Setelah model berhasil diunduh dan tersimpan dalam cache lokal, proses inferensi gambar dilakukan langsung di komputer pengguna, bukan melalui Serverless Cloud API.

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python 3.12 | Bahasa pemrograman utama |
| Roboflow Inference | Memuat dan menjalankan model pretrained |
| python-dotenv | Membaca konfigurasi dari file `.env` |
| Supervision | Mengolah hasil deteksi dan menggambar anotasi |
| OpenCV | Membaca, memproses, dan menyimpan gambar |
| YOLOv11 | Arsitektur object detection yang digunakan oleh model |

## 📂 Struktur Proyek

```text
Mango-Ripeness-Detection/
│
├── cek_model.py          # Memuat dan memeriksa model pretrained
├── uji_model.py          # Menjalankan deteksi kematangan mangga
├── images/               # Gambar input untuk inference
├── hasil_deteksi/        # Hasil anotasi, dibuat otomatis
├── requirements.txt      # Daftar dependensi Python
├── README.md             # Dokumentasi proyek
├── .gitignore            # Daftar file yang diabaikan Git
└── test.webp             # Contoh gambar untuk pengujian
```

**Catatan:**
- Direktori `.venv/` digunakan sebagai virtual environment dan tidak disertakan dalam repository.
- Letakkan gambar yang akan diprediksi di dalam folder `images/`. Script menerima JPG, JPEG, PNG, dan WebP.
- Folder `hasil_deteksi/` dibuat otomatis untuk menyimpan hasil prediksi dan dapat dipakai untuk melewati gambar yang sudah selesai diproses.
- `test.webp` tetap tersedia sebagai file contoh, tetapi `uji_model.py` saat ini membaca gambar dari folder `images/`.

## ⚙️ Instalasi dan Konfigurasi

### 1. Clone Repository

Unduh repository ke komputer menggunakan perintah:

```bash
git clone https://github.com/aqillazhr/Mango-Ripeness-Detection.git
cd Mango-Ripeness-Detection
```

### 2. Membuat Virtual Environment

Pastikan Python 3.12 telah terinstal.

**macOS / Linux:**

```bash
python3.12 -m venv .venv
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
```

Setelah virtual environment berhasil dibuat, aktifkan dengan perintah berikut.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Menginstal Dependencies

Pastikan virtual environment sudah aktif, kemudian jalankan:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Perintah tersebut akan menginstal seluruh library yang dibutuhkan berdasarkan `requirements.txt`.

Di Windows PowerShell, gunakan interpreter `.venv` secara langsung agar paket terpasang di environment proyek:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Konfigurasi Roboflow API Key

Proyek ini memerlukan Roboflow Private API Key untuk mengambil model pretrained dari Roboflow.

API Key dapat diperoleh melalui [Roboflow Dashboard](https://app.roboflow.com/).

Simpan API Key dan lokasi cache di file `.env` pada folder utama proyek. Script memuat file ini menggunakan `python-dotenv`.

```dotenv
ROBOFLOW_API_KEY=YOUR_PRIVATE_API_KEY
MODEL_CACHE_DIR=C:/Users/nama-pengguna/.cache/roboflow
INFERENCE_HOME=C:/Users/nama-pengguna/.cache/roboflow
```

Ganti `YOUR_PRIVATE_API_KEY` dan `nama-pengguna` dengan nilai milik Anda. Di macOS/Linux, sesuaikan path cache, misalnya `/home/nama-pengguna/.cache/roboflow`, untuk kedua variabel cache.

Pada Windows, buat folder cache jika belum ada:

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.cache\roboflow"
```

**Penting:** Jangan menuliskan Private API Key di kode atau mengunggah `.env` ke GitHub. File `.env` sudah diabaikan oleh `.gitignore`.

Pengaturan `MODEL_CACHE_DIR` dan `INFERENCE_HOME` pada `.env` membuat file model tersimpan di cache lokal agar dapat digunakan kembali pada eksekusi berikutnya.

## 🚀 Cara Menggunakan

### 1. Memeriksa Model

Sebelum menjalankan deteksi, pastikan model berhasil dimuat dengan menjalankan:

```bash
python cek_model.py
```

Script ini digunakan untuk memeriksa API Key, mengambil model pretrained apabila belum tersedia dalam cache, dan memastikan model berhasil dimuat.

### 2. Menyiapkan Gambar Input

Masukkan gambar yang ingin diprediksi ke folder `images/`. File contoh `test.webp` dapat digunakan dengan menyalinnya ke folder tersebut. Script memilih gambar berdasarkan urutan nama file dan hanya menggunakan format JPG, JPEG, PNG, atau WebP.

Jumlah gambar maksimal yang diproses pada satu kali eksekusi diatur lewat variabel berikut pada `uji_model.py`:

```python
IMAGE_DIR = Path("images")
NUM_IMAGES_TO_INFER = 2
```

Ubah `NUM_IMAGES_TO_INFER` untuk menentukan kuota gambar yang belum pernah diproses pada setiap eksekusi. Script melewati gambar jika hasilnya sudah ada di `hasil_deteksi/`, lalu mencari gambar berikutnya sampai kuota tercapai. Jika semua gambar sudah memiliki hasil, script selesai tanpa memuat model.

### 3. Menjalankan Deteksi

Setelah gambar tersedia di `images/` dan `.env` telah diisi, jalankan:

```bash
python uji_model.py
```

Secara umum, proses deteksi meliputi:

1. Memuat model pretrained CARABAO.
2. Memilih hingga `NUM_IMAGES_TO_INFER` gambar yang belum memiliki hasil.
3. Membaca gambar dari folder `images/` dan menjalankan object detection secara lokal.
4. Mengidentifikasi kelas kematangan dan confidence score.
5. Menggambar bounding box dan label pada objek yang terdeteksi.
6. Menyimpan setiap hasil anotasi di `hasil_deteksi/` dengan nama `<nama-gambar>_hasil.<ekstensi>`.

## 📊 Hasil Deteksi

Hasil deteksi berupa gambar yang telah diberi anotasi *bounding box*, label kelas kematangan, dan *confidence score* untuk setiap objek mangga yang berhasil dideteksi.

Contoh label prediksi:

- `Unripe`
- `Partially Ripe`
- `Ripe`

Gambar hasil deteksi disimpan terpisah di folder `hasil_deteksi/`. Jika file hasil untuk suatu gambar sudah ada, gambar input tersebut dilewati pada eksekusi berikutnya.

## 🔒 Catatan

- Koneksi internet dan Roboflow Private API Key diperlukan ketika model harus diambil dari Roboflow untuk pertama kali.
- Model yang telah diunduh disimpan dalam direktori cache lokal.
- Setelah model tersedia secara lokal, inferensi gambar dilakukan menggunakan Roboflow Inference tanpa mengirim gambar ke Serverless Cloud API.
- Direktori `.venv/`, API Key, cache lokal, dan file hasil deteksi tidak perlu diunggah ke repository.
- Hasil prediksi dapat dipengaruhi oleh kualitas gambar, pencahayaan, sudut pengambilan gambar, dan varietas mangga.
- Penggunaan model tetap mengikuti lisensi dan ketentuan atribusi dari sumber model aslinya.

## 📚 Referensi

- [CARABAO Ripeness Level – Roboflow Universe](https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me)
- [Roboflow Inference Documentation](https://inference.roboflow.com/)
- [Supervision Documentation](https://supervision.roboflow.com/)

## 📄 Atribusi Model

Model CARABAO Ripeness Level yang digunakan dalam proyek ini dipublikasikan oleh **MangoSort AI** melalui Roboflow Universe.

Model tersebut merupakan model pretrained dari pihak ketiga yang digunakan untuk kebutuhan implementasi dan pengujian deteksi kematangan mangga. Proyek ini tidak mengklaim kepemilikan atau pengembangan model tersebut.

Seluruh hak, lisensi, dan ketentuan penggunaan model mengikuti informasi yang ditetapkan oleh publisher pada halaman model aslinya.
