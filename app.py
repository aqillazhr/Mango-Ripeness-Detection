import os
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Model Roboflow parekh/mango-aiyie (Vision Transformer Classification)
MODEL_ID = "mango-aiyie/1"
MAX_IMAGES = 5
MAX_DISPLAY_WIDTH = 520
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

load_dotenv()

st.set_page_config(
    page_title="Mango Quality & Freshness Classification",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .badge-fresh {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-rotten {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .confidence-val {
        font-family: monospace;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    .status-box {
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .status-fresh {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #86efac;
    }
    .status-rotten {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
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
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key,
    )
    client.configure(InferenceConfiguration(api_key_transport="header"))
    return client


def resize_for_display(image: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        return cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return image


def annotate_classification_image(
    image: np.ndarray, top_class: str, confidence: float
) -> np.ndarray:
    """Menambahkan overlay banner status klasifikasi pada citra."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    banner_height = max(38, int(h * 0.08))
    font_scale = max(0.55, banner_height / 52.0)
    thickness = max(1, int(font_scale * 2))

    is_fresh = "fresh" in top_class.lower()
    banner_color = (34, 180, 70) if is_fresh else (40, 40, 220)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    text = f"{top_class} ({confidence * 100:.1f}%)"
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_x = 15
    text_y = int((banner_height + text_h) / 2)

    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


st.markdown(
    """
    <div class="header-container">
        <h1 class="title-text">Mango Quality & Freshness Classification</h1>
        <p class="subtitle-text">Klasifikasi kondisi buah mangga (Fresh Mango vs Rotten Mango) — maksimal 5 gambar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    st.error("Variabel ROBOFLOW_API_KEY tidak ditemukan di file .env")
    st.stop()

with st.spinner("Menghubungkan ke model Roboflow..."):
    client = load_detection_model()

if client is None:
    st.error("Gagal memuat klien model Roboflow.")
    st.stop()

with st.sidebar:
    st.subheader("⚙️ Informasi & Pengaturan")
    st.markdown(
        f"""
        **Model ID:** `{MODEL_ID}`  
        **Tipe:** *Classification (ViT)*  
        **Dataset:** [parekh/mango-aiyie](https://universe.roboflow.com/parekh/mango-aiyie)  
        **Kelas:** `Fresh Mango`, `Rotten Mango`
        """
    )
    st.markdown("---")
    min_confidence = st.slider(
        "Threshold Keyakinan Minimum",
        min_value=0.50,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Batas minimum keyakinan untuk menyimpulkan kelas buah secara definitif.",
    )
    st.caption("ℹ️ *Model mengklasifikasikan kondisi mangga segar (Fresh) atau busuk (Rotten).*")

uploaded_files = st.file_uploader(
    f"Unggah citra mangga (maksimal {MAX_IMAGES} berkas)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help=f"Maksimal {MAX_IMAGES} berkas sekaligus, maksimal {MAX_FILE_SIZE_MB}MB per berkas.",
)

if uploaded_files:
    if len(uploaded_files) > MAX_IMAGES:
        st.warning(
            f"Jumlah berkas terpilih ({len(uploaded_files)}) melebihi batas. "
            f"Hanya {MAX_IMAGES} gambar pertama yang akan diproses."
        )
        target_files = uploaded_files[:MAX_IMAGES]
    else:
        target_files = uploaded_files

    for idx, uploaded_file in enumerate(target_files, start=1):
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.error(
                f"[{idx}] {uploaded_file.name} dilewati: ukuran berkas ({size_mb:.2f}MB) "
                f"melebihi batas {MAX_FILE_SIZE_MB}MB."
            )
            st.markdown(
                "<hr style='border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0;'>",
                unsafe_allow_html=True,
            )
            continue

        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner(f"Memproses gambar {idx}/{len(target_files)}: {uploaded_file.name}..."):
            try:
                res = client.infer(cv_img, model_id=MODEL_ID)
                predictions = res.get("predictions", {})
                predicted_classes = res.get("predicted_classes", [])

                if predicted_classes:
                    top_class = predicted_classes[0]
                    top_conf = predictions.get(top_class, {}).get("confidence", 0.0)
                else:
                    sorted_preds = sorted(
                        predictions.items(),
                        key=lambda item: item[1].get("confidence", 0),
                        reverse=True,
                    )
                    top_class = sorted_preds[0][0] if sorted_preds else "Unknown"
                    top_conf = sorted_preds[0][1].get("confidence", 0.0) if sorted_preds else 0.0

                annotated = annotate_classification_image(cv_img, top_class, top_conf)
                annotated_resized = resize_for_display(annotated, max_width=MAX_DISPLAY_WIDTH)
                annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
            except Exception as e:
                st.error(f"Gagal memproses gambar {uploaded_file.name}: {e}")
                continue

        st.markdown(f'<div class="file-label">[{idx}] {uploaded_file.name}</div>', unsafe_allow_html=True)
        col_img, col_preds = st.columns([1, 1], gap="medium")

        with col_img:
            st.image(annotated_rgb, use_container_width=False)

        with col_preds:
            is_fresh = "fresh" in top_class.lower()
            badge_class = "badge-fresh" if is_fresh else "badge-rotten"
            status_style = "status-fresh" if is_fresh else "status-rotten"

            if top_conf < min_confidence:
                desc = (
                    f"⚠️ Keyakinan model ({top_conf * 100:.1f}%) berada di bawah threshold "
                    f"({min_confidence * 100:.0f}%). Hasil prediksi mungkin kurang meyakinkan."
                )
            elif is_fresh:
                desc = "✅ Buah terdeteksi dalam kondisi <strong>SEGAR</strong>. Kualitas prima dan layak untuk dikonsumsi maupun dipasarkan."
            else:
                desc = "⚠️ Buah terindikasi mengalami <strong>PEMBUSUKAN (Rotten)</strong> atau kerusakan kualitas. Sebaiknya disortir atau diperiksa lebih lanjut."

            st.markdown(
                f"""
                <div class="status-box {status_style}">
                    {desc}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("Distribusi Probabilitas Kelas:")
            # Urutkan prediksi berdasarkan confidence tertinggi
            sorted_classes = sorted(
                predictions.items(),
                key=lambda item: item[1].get("confidence", 0),
                reverse=True,
            )

            for c_idx, (cls_name, info) in enumerate(sorted_classes, start=1):
                conf = info.get("confidence", 0.0)
                cls_badge = "badge-fresh" if "fresh" in cls_name.lower() else "badge-rotten"
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div>
                            <span class="pred-class">#{c_idx} {cls_name}</span>
                            <span class="{cls_badge}">{cls_name}</span>
                        </div>
                        <div style="text-align: right;">
                            <span class="confidence-val">{conf * 100:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(max(float(conf), 0.0), 1.0))

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0;'>",
            unsafe_allow_html=True,
        )
