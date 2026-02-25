"""
Streamlit dashboard: upload a skin image → model predicts Leprosy / Not Leprosy + confidence.
Modern healthcare-themed UI with metric-focused layout.
"""
import streamlit as st
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "runs" / "detect" / "leprosy" / "weights" / "best.pt"
CONF_THRESHOLD = 0.25  # minimum confidence to count as "Leprosy"

# ─── Healthcare dashboard theme ─────────────────────────────────────────────
HEALTHCARE_CSS = """
<style>
  /* Root: clinical, clean palette */
  :root {
    --primary: #0d9488;
    --primary-dark: #0f766e;
    --primary-light: #5eead4;
    --surface: #f0fdfa;
    --card: #ffffff;
    --text: #134e4a;
    --text-muted: #5eead4;
    --success: #059669;
    --warning: #d97706;
    --border: #99f6e4;
    --shadow: 0 4px 20px rgba(13, 148, 136, 0.08);
    --radius: 12px;
    --radius-lg: 16px;
  }

  /* Hide default Streamlit chrome for cleaner look */
  #MainMenu, header[data-testid="stHeader"] { background: transparent !important; }
  .stDeployButton { display: none; }

  /* Page background */
  .stApp {
    background: #ffffff !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }

  /* Dashboard header */
  .dashboard-header {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    color: white;
    padding: 1.5rem 1.75rem;
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
  }
  .dashboard-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .dashboard-header p {
    margin: 0.35rem 0 0 0;
    opacity: 0.92;
    font-size: 0.9rem;
  }

  /* Metric cards */
  .metric-card {
    background: var(--card);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    margin-bottom: 1rem;
  }
  .metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0f766e;
    font-weight: 600;
    margin-bottom: 0.25rem;
  }
  .metric-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #134e4a;
    letter-spacing: -0.02em;
  }
  .metric-card .value.positive { color: #059669; }
  .metric-card .value.attention { color: #d97706; }

  /* Status badge */
  .status-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .status-badge.negative {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
  }
  .status-badge.positive {
    background: #d1fae5;
    color: #065f46;
    border: 1px solid #6ee7b7;
  }

  /* Confidence bar */
  .confidence-bar-wrap {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.5rem;
  }
  .confidence-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #0d9488, #5eead4);
    transition: width 0.4s ease;
  }

  /* Upload zone card */
  .upload-zone {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
  }

  /* Section titles */
  .section-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #0f766e;
    margin-bottom: 0.75rem;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown { color: #134e4a; }
</style>
"""


@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.exists():
        return None
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS_PATH))


def predict(image_path: str):
    model = load_model()
    if model is None:
        return None, "Model not found. Train first: python3 scripts/2_train.py"
    results = model.predict(source=image_path, conf=0.15, verbose=False)
    return results, None


def main():
    st.set_page_config(
        page_title="Leprosy Detection | Healthcare Screening",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(HEALTHCARE_CSS, unsafe_allow_html=True)

    # ─── Sidebar: context & info ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🩺 Screening Assistant")
        st.markdown("AI-assisted skin lesion screening. Upload an image to receive a **Leprosy** / **Not Leprosy** result with confidence.")
        st.markdown("---")
        st.markdown("**Threshold**  \nLesion confidence ≥ 25% → **Leprosy**")
        st.markdown("---")
        if load_model() is not None:
            st.success("Model loaded")
        else:
            st.error("Model not found")

    # ─── Main: header ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="dashboard-header">'
        '<h1>Leprosy Skin Lesion Detection</h1>'
        '<p>Upload a skin image for AI-assisted screening. Results include prediction and confidence.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if load_model() is None:
        st.error(f"Trained weights not found at `{WEIGHTS_PATH}`. Run training first: `python3 scripts/2_train.py`")
        return

    # ─── Upload ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Upload skin image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose file", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")
    if uploaded is None:
        st.info("Upload an image above to get a prediction.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
        f.write(uploaded.getvalue())
        tmp_path = f.name

    # ─── Inference ──────────────────────────────────────────────────────────
    with st.spinner("Running model..."):
        results, err = predict(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if err:
        st.error(err)
        return

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        label = "Not Leprosy"
        confidence = 0.0
    else:
        confs = boxes.conf.cpu().numpy()
        max_conf = float(confs.max())
        if max_conf >= CONF_THRESHOLD:
            label = "Leprosy"
            confidence = max_conf
        else:
            label = "Not Leprosy"
            confidence = max_conf

    # ─── Dashboard layout: image + metrics ───────────────────────────────────
    col_img, col_metrics = st.columns([1.2, 1])
    with col_img:
        st.markdown('<p class="section-title">Input image</p>', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)

    with col_metrics:
        st.markdown('<p class="section-title">Screening result</p>', unsafe_allow_html=True)
        badge_class = "negative" if label == "Leprosy" else "positive"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Prediction</div>'
            f'<div class="value"><span class="status-badge {badge_class}">{label}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        value_class = "attention" if label == "Leprosy" else "positive"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Confidence</div>'
            f'<div class="value {value_class}">{confidence:.1%}</div>'
            f'<div class="confidence-bar-wrap"><div class="confidence-bar-fill" style="width:{min(100, confidence*100)}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if label == "Leprosy":
            st.success(f"Lesion(s) detected with {confidence:.1%} confidence. Consider clinical follow-up.")
        else:
            st.info("No lesion detected above threshold.")

    # ─── Detection overlay (if any boxes) ────────────────────────────────────
    if boxes is not None and len(boxes) > 0 and hasattr(r, "plot"):
        st.markdown("---")
        st.markdown('<p class="section-title">Detection overlay</p>', unsafe_allow_html=True)
        import cv2
        plotted = r.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        _, col_overlay, _ = st.columns([1, 2, 1])  # center column = 50% width
        with col_overlay:
            st.image(plotted_rgb, caption="Detected regions", use_container_width=True)


if __name__ == "__main__":
    main()
