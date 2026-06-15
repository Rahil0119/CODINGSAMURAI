import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("✍️🔎 Handwritten Digit Classifier")
st.write("Test the neural network by either drawing a digit or uploading an image file!")

# 2. Load the Pre-trained CNN Model
@st.cache_resource
def load_mnist_model():
    return tf.keras.models.load_model('mnist_cnn.keras')

try:
    model = load_mnist_model()
except Exception as e:
    st.error("Could not load 'mnist_cnn.keras'. Make sure it's in the same folder as app.py")
    st.stop()

# 3. Create Frontend Navigation Tabs
tab1, tab2 = st.tabs(["🖌️ Draw Digit", "📤 Upload Digit"])
final_tensor = None

# --- TAB 1: DRAW DIGIT ---
with tab1:
    st.subheader("Draw a single digit inside the box")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=14,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None and np.any(canvas_result.image_data > 0):
        # Process canvas data (RGBA to Grayscale)
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img_resized = img.resize((28, 28))
        img_array = np.array(img_resized) / 255.0
        final_tensor = img_array.reshape(1, 28, 28, 1)

# --- TAB 2: UPLOAD DIGIT ---
with tab2:
    st.subheader("Upload an image file")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Open the uploaded image
        raw_img = Image.open(uploaded_file)
        
        # Display the uploaded image side-by-side with a small preview
        st.image(raw_img, caption="Uploaded Image", width=150)
        
        # Image Pipeline Preprocessing:
        # 1. Convert to Grayscale
        img_gray = raw_img.convert('L')
        
        # IMPORTANT MNIST RULE: Check if background is white and text is dark. 
        # MNIST is trained on white digits on a black background. If inverted, we fix it.
        img_np = np.array(img_gray)
        if np.mean(img_np) > 127:  
            img_gray = ImageOps.invert(img_gray)
            
        # 2. Resize to matching 28x28 dimension
        img_resized = img_gray.resize((28, 28))
        
        # 3. Normalize intensity values (0 to 1)
        img_array = np.array(img_resized) / 255.0
        
        # 4. Target input tensor structure shape
        final_tensor = img_array.reshape(1, 28, 28, 1)

# 4. Shared Model Inference Engine Block
if final_tensor is not None:
    # Run the prediction
    prediction = model.predict(final_tensor, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # 5. Display UI Predictions
    st.write("---")
    st.subheader("Model Inference Output")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Digit", value=str(predicted_digit))
    with col2:
        st.metric(label="Network Confidence", value=f"{confidence:.2f}%")
        
    st.bar_chart(prediction[0])