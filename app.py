import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import pickle

# Page settings
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    """Load the trained model (.keras or fallback to .json + .pkl)"""
    try:
        model_path = 'models/brain_tumor_cnn_classifier.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False)
            return model

        # Fallback: Load model from JSON + PKL
        with open('models/CNN_structure.json', 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        with open('models/CNN_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        model.set_weights(weights)
        return model

    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def preprocess_image(image):
    """Resize and normalize image"""
    image = image.resize((224, 224))
    image_array = np.array(image)
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    image_array = image_array.astype('float32') / 255.0
    return np.expand_dims(image_array, axis=0)

def main():
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    st.subheader("📁 Upload Brain MRI Scan")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded MRI", use_column_width=True)
        with col2:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size} bytes")
            st.write(f"**Dimensions:** {image.size}")
            st.write(f"**Mode:** {image.mode}")

        if st.button("🔬 Analyze"):
            with st.spinner("Analyzing..."):
                try:
                    image_array = preprocess_image(image)
                    predictions = model.predict(image_array)[0]
                    pred_idx = np.argmax(predictions)
                    pred_class = class_names[pred_idx]
                    confidence = predictions[pred_idx]

                    st.markdown("### 🎯 Result")
                    st.success(f"**Prediction:** {pred_class}")
                    st.info(f"**Confidence:** {confidence:.2%}")

                    # Plot
                    fig, ax = plt.subplots()
                    bars = ax.bar(class_names, predictions, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
                    for i, bar in enumerate(bars):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{predictions[i]:.2f}",
                                ha='center', va='bottom')
                    ax.set_ylim([0, 1])
                    ax.set_ylabel("Confidence")
                    st.pyplot(fig)

                    # Info
                    st.markdown("### 📋 Tumor Information")
                    tumor_info = {
                        "Glioma": "Tumor from glial cells. Often aggressive.",
                        "Meningioma": "Tumor from meninges. Often benign.",
                        "Pituitary": "Affects hormone regulation.",
                        "No Tumor": "No abnormalities detected."
                    }
                    st.warning(tumor_info[pred_class] if pred_class != "No Tumor" else "✅ No Tumor Detected")

                    st.markdown("""---""")
                    st.markdown("""
                    <div style="background:#f8d7da; padding:1rem; border-left:5px solid #dc3545;">
                        <strong>⚠️ Medical Disclaimer:</strong><br>
                        This tool is for educational use only. Consult professionals for actual diagnosis.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
