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
| Supervision | Mengolah hasil deteksi dan menggambar anotasi |
| OpenCV | Membaca, memproses, dan menyimpan gambar |
| YOLOv11 | Arsitektur object detection yang digunakan oleh model |

## 📂 Struktur Proyek

```text
Mango-Ripeness-Detection/
│
├── cek_model.py          # Memuat dan memeriksa model pretrained
├── uji_model.py          # Menjalankan deteksi kematangan mangga
├── requirements.txt      # Daftar dependensi Python
├── README.md             # Dokumentasi proyek
├── .gitignore            # Daftar file yang diabaikan Git
└── test.webp             # Contoh gambar untuk pengujian
```

**Catatan:**
- Direktori `.venv/` digunakan sebagai virtual environment dan tidak disertakan dalam repository.
- File `hasil_deteksi.jpg` akan dibuat secara otomatis setelah proses deteksi dijalankan dan tidak disertakan dalam repository.
- File `test.webp` disediakan sebagai gambar contoh agar pengguna dapat langsung mencoba model tanpa harus menyiapkan gambar sendiri.

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
pip install -r requirements.txt
```

Perintah tersebut akan menginstal seluruh library yang dibutuhkan berdasarkan `requirements.txt`.

### 4. Konfigurasi Roboflow API Key

Proyek ini memerlukan Roboflow Private API Key untuk mengambil model pretrained dari Roboflow.

API Key dapat diperoleh melalui [Roboflow Dashboard](https://app.roboflow.com/).

Atur API Key sebagai environment variable agar dapat digunakan oleh aplikasi.

**macOS / Linux:**

```bash
export ROBOFLOW_API_KEY="YOUR_PRIVATE_API_KEY"
```

**Windows (PowerShell):**

```powershell
$env:ROBOFLOW_API_KEY="YOUR_PRIVATE_API_KEY"
```

Ganti `YOUR_PRIVATE_API_KEY` dengan Private API Key milik masing-masing pengguna.

**Penting:** Jangan menuliskan Private API Key secara langsung di dalam kode atau mengunggahnya ke GitHub. Environment variable perlu diatur kembali ketika membuka sesi terminal baru.

### 5. Konfigurasi Model Cache

Model yang diunduh akan disimpan dalam direktori cache lokal sehingga dapat digunakan kembali pada eksekusi berikutnya.

**macOS / Linux:**

```bash
export MODEL_CACHE_DIR="$HOME/.cache/roboflow"
export INFERENCE_HOME="$MODEL_CACHE_DIR"
mkdir -p "$MODEL_CACHE_DIR"
```

**Windows (PowerShell):**

```powershell
$env:MODEL_CACHE_DIR="$HOME\.cache\roboflow"
$env:INFERENCE_HOME=$env:MODEL_CACHE_DIR
New-Item -ItemType Directory -Force -Path $env:MODEL_CACHE_DIR
```

Konfigurasi cache ini membantu mempertahankan file model yang telah diunduh sehingga tidak perlu mengunduhnya kembali setiap kali aplikasi dijalankan, selama file cache masih tersedia.

## 🚀 Cara Menggunakan

### 1. Memeriksa Model

Sebelum menjalankan deteksi, pastikan model berhasil dimuat dengan menjalankan:

```bash
python cek_model.py
```

Script ini digunakan untuk memeriksa API Key, mengambil model pretrained apabila belum tersedia dalam cache, dan memastikan model berhasil dimuat.

### 2. Menyiapkan Gambar Input

Repository ini sudah menyediakan gambar contoh bernama `test.webp` yang dapat langsung digunakan untuk mencoba deteksi kematangan mangga.

Pastikan konfigurasi pada `uji_model.py` menggunakan:

```python
IMAGE_PATH = "test.webp"
```

Dengan konfigurasi tersebut, pengguna dapat langsung menjalankan pengujian menggunakan gambar yang telah disediakan tanpa perlu menyiapkan gambar tambahan.

**Menggunakan Gambar Sendiri**

Pengguna juga dapat menggunakan gambar mangga milik sendiri sebagai input.

Langkah-langkahnya:

1. Letakkan gambar yang ingin diuji di dalam direktori proyek.
2. Buka file `uji_model.py`.
3. Ubah nilai `IMAGE_PATH` sesuai dengan nama file gambar yang digunakan.

Sebagai contoh, apabila gambar yang ingin digunakan bernama `mango_sample.jpg`, ubah konfigurasi menjadi:

```python
IMAGE_PATH = "mango_sample.jpg"
```

Pengguna juga dapat menggunakan gambar dengan format lain, misalnya:

```python
IMAGE_PATH = "test2.avif"
```

Beberapa format gambar yang dapat digunakan dalam proyek ini meliputi JPG, JPEG, PNG, WebP, dan AVIF. Dukungan format bergantung pada kemampuan decoding OpenCV yang terpasang di lingkungan pengguna.

Dengan demikian, pengguna bebas menggunakan gambar contoh yang tersedia maupun menggantinya dengan gambar mangga sendiri.

### 3. Menjalankan Deteksi

Setelah gambar input ditentukan, jalankan perintah:

```bash
python uji_model.py
```

Secara umum, proses deteksi meliputi:

1. Memuat model pretrained CARABAO.
2. Membaca gambar input berdasarkan konfigurasi `IMAGE_PATH`.
3. Menjalankan object detection secara lokal.
4. Mengidentifikasi kelas kematangan yang diprediksi.
5. Mengambil confidence score hasil prediksi.
6. Menggambar bounding box dan label pada objek yang terdeteksi.
7. Menyimpan hasil anotasi sebagai `hasil_deteksi.jpg`.

## 📊 Hasil Deteksi

Hasil deteksi berupa gambar yang telah diberi anotasi *bounding box*, label kelas kematangan, dan *confidence score* untuk setiap objek mangga yang berhasil dideteksi.

Contoh label prediksi:

- `Unripe`
- `Partially Ripe`
- `Ripe`

Gambar hasil deteksi disimpan dengan nama `hasil_deteksi.jpg` di direktori proyek. File tersebut akan diperbarui ketika script dijalankan kembali.

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