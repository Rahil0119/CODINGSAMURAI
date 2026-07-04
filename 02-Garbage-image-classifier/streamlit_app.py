import streamlit as st
from PIL import Image
from openCV import init_model, predict_pil_image, dustbin_mapping, classes

st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="centered")

@st.cache_resource
def load_model():
    try:
        return init_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

st.title("🗑️ Garbage Classifier")
st.write("Upload an image or take a photo to identify the type of garbage.")

mode = st.radio("Choose input method", ["Upload Image", "Camera Capture"])
uploaded_image = None

if mode == "Upload Image":
    f = st.file_uploader("Upload a photo", type=["jpg","jpeg","png"])
    if f:
        uploaded_image = Image.open(f).convert("RGB")
else:
    cam = st.camera_input("Take a picture")
    if cam:
        uploaded_image = Image.open(cam).convert("RGB")

if uploaded_image and model:
    st.image(uploaded_image, caption="Input image", use_column_width=True)
    label, confidence, probabilities = predict_pil_image(uploaded_image, model)

    # ✅ Unpack 3-tuple: name, bgr (unused here), css string
    bin_name, _bgr, css_color = dustbin_mapping[label]

    st.markdown("---")
    st.subheader("Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Class",      label.upper())
    col2.metric("Confidence", f"{confidence:.1f}%")
    col3.metric("Bin",        bin_name)

    # ✅ Fixed: unsafe_allow_html=True + valid CSS color string
    st.markdown(
        f"<span style='color:{css_color}; font-size:28px;'>■</span>"
        f" <b style='color:{css_color};'>{bin_name}</b>",
        unsafe_allow_html=True
    )

    st.markdown("### All Class Probabilities")
    prob_dict = {classes[i]: float(f"{probabilities[i]*100:.1f}") for i in range(len(classes))}
    st.bar_chart(prob_dict)   # visual bar chart instead of raw JSON

    st.markdown("---")
    st.caption("Garbage classes: " + ", ".join(c.capitalize() for c in classes))