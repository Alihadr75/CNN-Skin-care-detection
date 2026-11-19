from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Lesion Image Classifier", page_icon="🔬", layout="centered")
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "model.h5"
CLASS_PATH = ROOT / "models" / "class_names.json"
IMG_SIZE = (224, 224)

# HAM10000 label descriptions. These are dataset categories, not diagnoses.
LABEL_INFO = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

st.title("🔬 Skin Lesion Image Classifier")
st.write("Upload a dermatoscopic skin-lesion image for an experimental 7-class CNN prediction.")

st.warning(
    "Educational research demonstration only. This model is not a medical device, "
    "cannot diagnose skin cancer, and must not be used to decide treatment or delay care. "
    "Image quality, skin tone, camera type, and dataset bias can affect predictions. "
    "Consult a qualified dermatologist for any concerning or changing lesion."
)

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train.py first and place model.h5 in the models/ folder."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    return model, class_names

try:
    model, class_names = load_assets()
except Exception as exc:
    st.error(str(exc))
    st.info("Train the model locally or in Google Colab, then add models/model.h5 to your project.")
    st.stop()

uploaded = st.file_uploader("Choose a skin-lesion image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Analyze image", type="primary", use_container_width=True):
        resized = image.resize(IMG_SIZE)
        array = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)
        probabilities = model.predict(batch, verbose=0)[0]
        order = np.argsort(probabilities)[::-1]
        best_idx = int(order[0])
        best_label = class_names[best_idx]
        st.subheader("Model output (not a diagnosis)")
        st.metric("Top predicted dataset category", best_label.upper())
        st.write(LABEL_INFO.get(best_label, "Dataset category"))
        st.caption(f"Model score: {probabilities[best_idx]:.1%} — this is not a calibrated probability of disease.")
        rows = [
            {"Class": class_names[i], "Dataset label meaning": LABEL_INFO.get(class_names[i], ""),
             "Model score": float(probabilities[i])}
            for i in order
        ]
        st.dataframe(rows, use_container_width=True)
        st.bar_chart({r["Class"]: r["Model score"] for r in rows})

st.caption("Dataset: HAM10000 (7 dermatoscopic lesion categories). Research/education use only.")
