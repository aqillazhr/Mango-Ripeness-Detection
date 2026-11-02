# 🥭 Mango Ripeness Detection

Aplikasi **Object Detection untuk mendeteksi tingkat kematangan buah mangga** menggunakan model YOLO dari Roboflow. Proyek ini menggunakan Roboflow Inference untuk menjalankan prediksi secara lokal di komputer.

## 📌 Project Overview

Proyek ini bertujuan untuk mendeteksi dan mengklasifikasikan tingkat kematangan buah mangga berdasarkan citra digital.

Model mendeteksi objek mangga sekaligus memberikan bounding box dan label kelas kematangan pada setiap objek yang teridentifikasi.

Pengujian dilakukan menggunakan model pretrained dari Roboflow Universe, yaitu **CARABAO Ripeness Level**.

## 🎯 Detection Classes

Model memiliki tiga kelas tingkat kematangan:

| Class | Description |
|---|---|
| `Unripe` | Mangga mentah |
| `Partially Ripe` | Mangga setengah matang |
| `Ripe` | Mangga matang |

## 🤖 Pretrained Model

Model yang digunakan:

- **Model Name:** CARABAO Ripeness Level
- **Model ID:** `carabao-ripeness-level-0x9me/19`
- **Architecture:** YOLOv11
- **Task:** Object Detection
- **Platform:** Roboflow Universe

**Model Source:** [CARABAO Ripeness Level – Roboflow Universe](https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me)

Model diunduh melalui Roboflow Inference dan disimpan dalam cache lokal. Setelah model tersedia, proses inferensi gambar dilakukan di komputer pengguna, bukan melalui Serverless Cloud API.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| Roboflow Inference | Loading and running the pretrained model |
| Supervision | Processing detection results and drawing bounding boxes |
| OpenCV | Image processing and saving detection results |
| YOLOv11 | Object detection architecture |

## 📂 Project Structure

```text
mangga/
│
├── .venv/                # Python virtual environment (not uploaded)
│
├── cek_model.py          # Load and verify the pretrained model
├── uji_model.py          # Run mango ripeness detection
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignored files
│
├── test.webp             # Example test image (optional)
└── hasil_deteksi.jpg     # Detection output (generated)
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mangga
```

### 2. Create a Virtual Environment

Make sure Python 3.12 is installed.

```bash
python3.12 -m venv .venv
```

Activate the virtual environment:

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Roboflow API Key

A Private API Key is required to retrieve the pretrained model from Roboflow.

Obtain your API Key from [Roboflow Dashboard](https://app.roboflow.com).

Set the API Key as an environment variable.

**macOS / Linux:**

```bash
export ROBOFLOW_API_KEY="YOUR_PRIVATE_API_KEY"
```

**Windows (PowerShell):**

```powershell
$env:ROBOFLOW_API_KEY="YOUR_PRIVATE_API_KEY"
```

Do not hardcode or upload your Private API Key to GitHub.

### 5. Configure Model Cache

To store downloaded model files in a persistent local directory:

**macOS / Linux:**

```bash
export MODEL_CACHE_DIR="$HOME/.cache/roboflow"
export INFERENCE_HOME="$MODEL_CACHE_DIR"
mkdir -p "$MODEL_CACHE_DIR"
```

## 🚀 Usage

### 1. Verify Model Loading

Run the following command:

```bash
python cek_model.py
```

This script checks the API Key, retrieves the pretrained model, and verifies that the model can be loaded successfully.

### 2. Prepare a Test Image

Place a mango image in the project directory.

Update the image filename in `uji_model.py`:

```python
IMAGE_PATH = "test.webp"
```

The application supports image formats readable by OpenCV, including JPG, PNG, and WebP.

### 3. Run Detection

Execute:

```bash
python uji_model.py
```

The program will:
1. Load the pretrained CARABAO model.
2. Read the input image.
3. Perform object detection locally.
4. Identify the predicted ripeness class.
5. Calculate the confidence score.
6. Draw bounding boxes and labels.
7. Save the annotated image as `hasil_deteksi.jpg`.

## 📊 Detection Output

The output image contains bounding boxes around detected mangoes, along with their predicted ripeness classes and confidence scores.

Example output labels:

- `Unripe`
- `Partially Ripe`
- `Ripe`

## 🔒 Notes

- An internet connection and Roboflow Private API Key are required when retrieving the model for the first time.
- Model files are stored in the local cache directory.
- Image inference is performed locally using Roboflow Inference.
- The `.venv` directory, API Key, local cache, and generated output files should not be committed to GitHub.
- Model predictions may vary depending on image quality, lighting, camera angle, and mango variety.

## 📚 References

- [CARABAO Ripeness Level – Roboflow Universe](https://universe.roboflow.com/mangosort-ai/carabao-ripeness-level-0x9me)
- [Roboflow Inference Documentation](https://inference.roboflow.com/)
- [Supervision Documentation](https://supervision.roboflow.com/)

## 📄 Model Attribution

The pretrained model is sourced from MangoSort AI through Roboflow Universe. Refer to the original model page for its license and attribution requirements.