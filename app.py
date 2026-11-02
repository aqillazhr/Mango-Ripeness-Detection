import os
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

# shortcut: model hardcoded ke carabao-ripeness-level-0x9me/19, ubah bila ada model baru
MODEL_ID = "carabao-ripeness-level-0x9me/19"
MAX_IMAGES = 5
MAX_DISPLAY_WIDTH = 520
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

load_dotenv()

st.set_page_config(
    page_title="Mango Ripeness Detection",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    .header-container {
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.5rem;
    }
    .title-text {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }
    .subtitle-text {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    .result-row {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .file-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
    }
    .prediction-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .pred-class {
        font-weight: 600;
        font-size: 0.9rem;
        color: #f8fafc;
    }
    .badge-ripe {
        background: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .badge-partially-ripe {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .badge-unripe {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .confidence-val {
        font-family: monospace;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_detection_model():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        return None
    return get_model(model_id=MODEL_ID, api_key=api_key)

def resize_for_display(image: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    # shortcut: hanya mengecilkan jika lebar melebihi batas, tanpa sharpening filter
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        return cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return image

st.markdown(
    """
    <div class="header-container">
        <h1 class="title-text">Mango Ripeness Detection</h1>
        <p class="subtitle-text">Deteksi kematangan mangga (Unripe, Partially Ripe, Ripe) — maksimal 5 gambar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    st.error("Variabel ROBOFLOW_API_KEY tidak ditemukan di .env")
    st.stop()

with st.spinner("Memuat model..."):
    model = load_detection_model()

if model is None:
    st.error("Gagal memuat model Roboflow.")
    st.stop()

uploaded_files = st.file_uploader(
    f"Unggah citra mangga (maksimal {MAX_IMAGES} berkas)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help=f"Maksimal {MAX_IMAGES} berkas sekaligus.",
)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.warning(f"Jumlah file terpilih ({len(uploaded_files)}) melebihi batas. Hanya {MAX_IMAGES} gambar pertama yang akan diproses.")
        target_files = uploaded_files[:MAX_IMAGES]
    else:
        target_files = uploaded_files

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for idx, uploaded_file in enumerate(target_files, start=1):
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.error(f"[{idx}] {uploaded_file.name} dilewati: ukuran file ({size_mb:.2f}MB) melebihi batas {MAX_FILE_SIZE_MB}MB.")
            st.markdown("<hr style='border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0;'>", unsafe_allow_html=True)
            continue

        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner(f"Memproses gambar {idx}/{len(target_files)}: {uploaded_file.name}..."):
            results = model.infer(cv_img)
            predictions_data = results[0]
            detections = sv.Detections.from_inference(predictions_data)

            labels = [
                f"{pred.class_name} {pred.confidence:.2f}"
                for pred in predictions_data.predictions
            ]

            annotated = box_annotator.annotate(scene=cv_img.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            # Resize agar tampilan proporsional dan tidak memakan layar
            annotated_resized = resize_for_display(annotated, max_width=MAX_DISPLAY_WIDTH)
            annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)

        st.markdown(f'<div class="file-label">[{idx}] {uploaded_file.name}</div>', unsafe_allow_html=True)
        col_img, col_preds = st.columns([1, 1], gap="medium")

        with col_img:
            st.image(annotated_rgb, use_container_width=False)

        with col_preds:
            preds = predictions_data.predictions
            if not preds:
                st.info("Tidak ada objek mangga yang terdeteksi.")
            else:
                st.caption(f"Terdeteksi {len(preds)} objek")
                for p_idx, p in enumerate(preds, start=1):
                    badge_class = "badge-unripe"
                    cls_lower = p.class_name.lower()
                    if "partially" in cls_lower:
                        badge_class = "badge-partially-ripe"
                    elif "ripe" in cls_lower:
                        badge_class = "badge-ripe"

                    conf_percent = p.confidence * 100
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div>
                                <span class="pred-class">#{p_idx} {p.class_name}</span>
                                <span class="{badge_class}">{p.class_name}</span>
                            </div>
                            <div style="text-align: right;">
                                <span class="confidence-val">{conf_percent:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(float(p.confidence), 0.0), 1.0))

        st.markdown("<hr style='border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0;'>", unsafe_allow_html=True)
