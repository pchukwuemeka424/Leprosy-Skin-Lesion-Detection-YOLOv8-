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

# ─── Dashboard theme: midnight blue + red, white text on backgrounds ─────────
HEALTHCARE_CSS = """
<style>
  :root {
    --midnight: #191970;
    --midnight-light: #2a2a8a;
    --red: #b91c1c;
    --red-light: #dc2626;
    --card: #ffffff;
    --text: #0f172a;
    --text-muted: #475569;
    --radius: 12px;
    --radius-lg: 16px;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
  }

  #MainMenu, header[data-testid="stHeader"] { background: transparent !important; }
  .stDeployButton { display: none; }

  /* Page: solid background */
  .stApp {
    background: #f1f5f9 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }

  /* Breadcrumb bar - midnight blue, white text */
  .breadcrumb {
    background: #191970;
    color: #ffffff;
    padding: 0.65rem 1.5rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin: -1rem -1rem 0 -1rem;
    border-radius: 0;
    box-shadow: var(--shadow-md);
  }
  .breadcrumb span { font-weight: 700; color: #ffffff; }

  /* Hero header - midnight blue, white text */
  .dashboard-header {
    background: #191970;
    color: #ffffff;
    padding: 2rem 2rem;
    border-radius: var(--radius-lg);
    margin: 1.25rem 0 1.25rem 0;
    box-shadow: var(--shadow-md);
    border: 1px solid #2a2a8a;
  }
  .dashboard-header h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
  }
  .dashboard-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    color: #ffffff;
    opacity: 0.95;
  }

  /* Instructions card - midnight blue background, white text */
  .instructions-card {
    background: #191970;
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-md);
    border: 1px solid #2a2a8a;
  }
  .instructions-card h3 {
    margin: 0 0 0.85rem 0;
    font-size: 1.05rem;
    color: #ffffff;
    font-weight: 700;
  }
  .instructions-card ol {
    margin: 0;
    padding-left: 1.35rem;
    color: #ffffff;
    font-size: 0.92rem;
    line-height: 1.7;
  }
  .instructions-card li { margin-bottom: 0.4rem; color: #ffffff; }
  .instructions-card .disclaimer {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.3);
    font-size: 0.82rem;
    color: #e2e8f0;
    font-style: italic;
  }

  /* Section titles - dark when on light, white when in colored section */
  .section-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #191970;
    margin-bottom: 0.6rem;
  }
  .upload-section .section-title,
  .overlay-section .section-title { color: #ffffff; }

  /* Upload section - midnight blue, white text */
  .upload-section {
    background: #191970;
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-md);
    border: 1px solid #2a2a8a;
    color: #ffffff;
  }
  .upload-section p { color: #ffffff !important; }

  /* Metric cards - white card, colored left border */
  .metric-card {
    background: #ffffff;
    border-radius: var(--radius);
    padding: 1.35rem 1.5rem;
    box-shadow: var(--shadow);
    border-left: 4px solid #191970;
    margin-bottom: 1rem;
  }
  .metric-card.prediction-positive { border-left-color: #191970; }
  .metric-card.prediction-attention { border-left-color: #b91c1c; }
  .metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 0.3rem;
  }
  .metric-card .value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
  }
  .metric-card .value.positive { color: #059669; }
  .metric-card .value.attention { color: #b91c1c; }

  .status-badge {
    display: inline-block;
    padding: 0.5rem 1.1rem;
    border-radius: 9999px;
    font-size: 0.92rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .status-badge.negative {
    background: #b91c1c;
    color: #ffffff;
    border: 1px solid #dc2626;
  }
  .status-badge.positive {
    background: #191970;
    color: #ffffff;
    border: 1px solid #2a2a8a;
  }

  .confidence-bar-wrap {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.6rem;
  }
  .confidence-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: #b91c1c;
    transition: width 0.4s ease;
  }

  /* Detection overlay section - midnight blue, white text */
  .overlay-section {
    background: #191970;
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-top: 1.25rem;
    border: 1px solid #2a2a8a;
    color: #ffffff;
  }
  .overlay-section .section-title { color: #ffffff; }
  .overlay-section p { color: #e2e8f0 !important; }

  /* Sidebar - midnight blue, white text */
  [data-testid="stSidebar"] {
    background: #191970 !important;
    border-right: 1px solid #2a2a8a;
  }
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label { color: #ffffff !important; }

  [data-testid="stAlert"] { border-radius: var(--radius); }
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

    # ─── Breadcrumb: owner ────────────────────────────────────────────────────
    st.markdown(
        '<div class="breadcrumb">'
        '<span>Owner</span> &nbsp;|&nbsp; <span>Name:</span> Fortune Chioma &nbsp;|&nbsp; <span>Matriculation No.:</span> 22/sci01/086'
        '</div>',
        unsafe_allow_html=True,
    )

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
        '<h1>🩺 Leprosy Skin Lesion Detection</h1>'
        '<p>AI-assisted screening: upload a skin image to get a Leprosy / Not Leprosy prediction with confidence.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if load_model() is None:
        st.error(f"Trained weights not found at `{WEIGHTS_PATH}`. Run training first: `python3 scripts/2_train.py`")
        return

    # ─── Upload ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="upload-section">'
        '<p class="section-title">Step 1 — Upload skin image</p>'
        '<p style="margin:0; color:#ffffff; font-size:0.9rem;">Choose an image of the skin area to screen. Supported: JPG, PNG, WebP, BMP.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Choose file", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")
    if uploaded is None:
        st.info("👆 **Upload an image** above to get started. Supported formats: JPG, PNG, WebP, BMP.")
        st.markdown(
            '<div class="instructions-card">'
            '<h3>📋 How to use</h3>'
            '<ol>'
            '<li><strong>Upload</strong> a skin image using the file uploader above (JPG, PNG, WebP, or BMP).</li>'
            '<li><strong>Wait</strong> a few seconds while the AI model analyzes the image.</li>'
            '<li><strong>Review</strong> the prediction (Leprosy or Not Leprosy) and the confidence score.</li>'
            '<li>If lesions are detected, check the <strong>Detection overlay</strong> to see highlighted regions.</li>'
            '<li>Use the sidebar to see the detection threshold (≥25% confidence → Leprosy).</li>'
            '</ol>'
            '<p class="disclaimer">This tool is for screening support only. Always seek professional medical advice for diagnosis and treatment.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
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

    # ─── One row: Screening result | Result | Detection overlay ───────────────
    has_overlay = boxes is not None and len(boxes) > 0 and hasattr(r, "plot")
    if has_overlay:
        import cv2
        plotted = r.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    else:
        plotted_rgb = None

    col_screening, col_result, col_overlay = st.columns(3)
    with col_screening:
        st.markdown('<p class="section-title">Screening result</p>', unsafe_allow_html=True)
        st.image(uploaded, caption="Your uploaded image", width=280)

    with col_result:
        st.markdown('<p class="section-title">Result</p>', unsafe_allow_html=True)
        badge_class = "negative" if label == "Leprosy" else "positive"
        card_class_pred = "prediction-attention" if label == "Leprosy" else "prediction-positive"
        st.markdown(
            f'<div class="metric-card {card_class_pred}">'
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
            st.success(f"**Lesion(s) detected** ({confidence:.1%} confidence). Consider clinical follow-up.")
        else:
            st.info("**No lesion detected** above the 25% threshold.")

    with col_overlay:
        st.markdown('<p class="section-title">Detection overlay</p>', unsafe_allow_html=True)
        if plotted_rgb is not None:
            st.image(plotted_rgb, caption="Detected regions", width=280)
        else:
            st.caption("No regions detected above threshold.")


if __name__ == "__main__":
    main()
