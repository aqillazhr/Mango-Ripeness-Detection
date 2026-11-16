import os
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import supervision as sv
from ultralytics import YOLO

# Konfigurasi Model Lokal (best_v2.pt)
MODEL_CANDIDATES = [
    Path("runs/detect/runs/mango/yolo11n_mango/weights/best_v2.pt"),
    Path("runs/mango/yolo11n_mango/weights/best_v2.pt"),
    Path("runs/mango/weights/best_v2.pt"),
    Path("best_v2.pt"),
    Path("runs/detect/runs/mango/yolo11n_mango/weights/best.pt"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)
MAX_IMAGES = 5
MAX_DISPLAY_WIDTH = 520
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(
    page_title="Mango Ripeness Detection (Local YOLO11)",
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
    .badge-ripe {
        background: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-partially-ripe {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-unripe {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
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
    .status-ripe {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.25);
        color: #fde047;
    }
    .status-partially-ripe {
        background: rgba(249, 115, 22, 0.1);
        border: 1px solid rgba(249, 115, 22, 0.25);
        color: #fdba74;
    }
    .status-unripe {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #86efac;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_detection_model(model_path: str):
    return YOLO(model_path)


def resize_for_display(image: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
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
        <p class="subtitle-text">Deteksi dan klasifikasi tingkat kematangan mangga (Unripe, Partially Ripe, Ripe) — Model Lokal YOLOv11.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if MODEL_PATH is None or not MODEL_PATH.exists():
    st.error(
        "❌ **File model lokal (`best_v2.pt`) tidak ditemukan!**\n\n"
        "Pastikan file `best_v2.pt` telah diletakkan di direktori `runs/` atau folder proyek."
    )
    st.stop()

with st.spinner("Memuat model YOLO lokal (best_v2.pt)..."):
    try:
        model = load_detection_model(str(MODEL_PATH))
    except Exception as e:
        st.error(f"Gagal memuat model lokal: {e}")
        st.stop()

with st.sidebar:
    st.subheader("⚙️ Informasi & Pengaturan")
    st.markdown(
        f"""
        **Model:** `Local YOLO11 ({MODEL_PATH.name})`  
        **Path:** `{MODEL_PATH}`  
        **Tipe:** *Object Detection (YOLOv11)*  
        **Kelas:** `Partially Ripe` , `Ripe` , `Unripe`  
        **Status:** 🟢 *Model Lokal Siap (Offline)*
        """
    )
    st.markdown("---")
    st.markdown("##### 🎛️ Pengaturan Deteksi")
    conf_threshold = st.slider(
        "Threshold Keyakinan (Confidence)",
        min_value=0.05,
        max_value=1.00,
        value=0.20,
        step=0.05,
        help="Batas minimal keyakinan deteksi. Turunkan nilai ini (misal 0.15) jika ada mangga yang terlewat.",
    )
    iou_threshold = st.slider(
        "Threshold NMS (IoU)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Mengatasi kotak ganda (duplikat). Semakin kecil nilainya, semakin agresif menghapus kotak bertumpuk pada 1 mangga.",
    )
    agnostic_nms = st.toggle(
        "Hapus Kotak Ganda Antar-Kelas (Agnostic NMS)",
        value=True,
        help="Mencegah 1 mangga memiliki 2 bounding box berbeda kelas secara bersamaan.",
    )
    st.caption("ℹ️ *Tips: Jika ada mangga terlewat, turunkan Confidence ke 0.15. Jika 1 mangga punya 2 kotak, aktifkan Agnostic NMS dan setel IoU ke 0.40 - 0.45.*")

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

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

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
                results = model.predict(
                    source=cv_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    agnostic_nms=agnostic_nms,
                    imgsz=640,
                    verbose=False,
                )[0]
                detections = sv.Detections.from_ultralytics(results)

                labels = [
                    f"{model.names[cls_id]} {conf:.2f}"
                    for cls_id, conf in zip(detections.class_id, detections.confidence)
                ]

                annotated = box_annotator.annotate(scene=cv_img.copy(), detections=detections)
                annotated = label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )

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
            preds = [
                {
                    "class_name": model.names[cls_id],
                    "confidence": float(conf),
                }
                for cls_id, conf in zip(detections.class_id, detections.confidence)
            ]

            if not preds:
                st.info("Tidak ada objek mangga yang terdeteksi dengan threshold keyakinan saat ini.")
            else:
                # Hitung ringkasan kelas
                classes_count = {}
                for p in preds:
                    cname = p["class_name"]
                    classes_count[cname] = classes_count.get(cname, 0) + 1

                dominant_class = max(classes_count, key=classes_count.get)
                dom_lower = dominant_class.lower()

                if "un" in dom_lower:
                    status_style = "status-unripe"
                    desc = f"Terdeteksi <strong>{len(preds)} objek mangga</strong>. Mayoritas berada pada tingkat <strong>Unripe (Mentah)</strong>."
                elif "semi" in dom_lower or "partially" in dom_lower:
                    status_style = "status-partially-ripe"
                    desc = f"Terdeteksi <strong>{len(preds)} objek mangga</strong>. Mayoritas berada pada tingkat <strong>Partially Ripe (Setengah Matang)</strong>."
                else:
                    status_style = "status-ripe"
                    desc = f"Terdeteksi <strong>{len(preds)} objek mangga</strong>. Mayoritas berada pada tingkat <strong>Ripe (Matang)</strong>."

                st.markdown(
                    f"""
                    <div class="status-box {status_style}">
                        {desc}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(f"Daftar Objek Terdeteksi ({len(preds)} buah):")

                # Urutkan berdasarkan confidence tertinggi
                sorted_preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)

                for p_idx, p in enumerate(sorted_preds, start=1):
                    cls_name = p["class_name"]
                    conf_val = p["confidence"]
                    cls_lower = cls_name.lower()

                    if "un" in cls_lower:
                        badge_class = "badge-unripe"
                    elif "semi" in cls_lower or "partially" in cls_lower:
                        badge_class = "badge-partially-ripe"
                    elif "ripe" in cls_lower:
                        badge_class = "badge-ripe"
                    else:
                        badge_class = "badge-unripe"

                    conf_percent = conf_val * 100
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div>
                                <span class="pred-class">#{p_idx} {cls_name}</span>
                                <span class="{badge_class}">{cls_name}</span>
                            </div>
                            <div style="text-align: right;">
                                <span class="confidence-val">{conf_percent:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(max(float(conf_val), 0.0), 1.0))

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0;'>",
            unsafe_allow_html=True,
        )
