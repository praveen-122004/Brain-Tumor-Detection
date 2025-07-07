import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import pickle

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Global variable to store the model
@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    try:
        st.info("Loading model... Please wait.")
        
        # Try to load the complete model first
        model_path = 'models/brain_tumor_cnn_classifier.keras'
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                st.success("✅ Model loaded successfully!")
                return model
            except Exception as e:
                st.warning(f"Could not load complete model: {str(e)}")
        
        # Fallback: load from structure and weights
        structure_path = 'models/CNN_structure.json'
        weights_path = 'models/CNN_weights.pkl'
        
        if not os.path.exists(structure_path):
            st.error(f"Model structure file not found: {structure_path}")
            return None
            
        if not os.path.exists(weights_path):
            st.error(f"Model weights file not found: {weights_path}")
            return None
        
        # Load model structure
        with open(structure_path, 'r') as json_file:
            model_json = json_file.read()
        
        model = tf.keras.models.model_from_json(model_json)
        
        # Load weights
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        
        # Set weights
        model.set_weights(weights)
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        st.success("✅ Model loaded successfully from structure and weights!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert to array and normalize
    image_array = np.array(image)
    # Ensure 3 channels (RGB)
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:, :, :3]
    
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check your model files.")
        st.stop()
    
    # Class names
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    st.markdown("### 📁 Upload Brain MRI Scan")
    
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan in JPG, PNG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, caption="Uploaded Brain MRI Scan", use_column_width=True)
        
        with col2:
            st.markdown("### 📊 Image Information")
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
            st.write(f"**Image dimensions:** {image.size}")
            st.write(f"**Image mode:** {image.mode}")
        
        # Analysis button
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing brain scan..."):
                try:
                    # Preprocess image
                    image_array = preprocess_image(image)
                    
                    # Get prediction
                    predictions = model.predict(image_array, verbose=0)
                    predictions = predictions[0]
                    
                    # Get predicted class
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = class_names[predicted_class_idx]
                    confidence = predictions[predicted_class_idx]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Analysis Results")
                    
                    # Prediction box
                    st.success(f"**Prediction: {predicted_class}**")
                    st.info(f"**Confidence: {confidence:.3f} ({confidence*100:.1f}%)**")
                    
                    # Confidence plot
                    st.markdown("### 📈 Prediction Confidence")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(class_names, predictions, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
                    
                    # Add value labels on bars
                    for i, (bar, pred) in enumerate(zip(bars, predictions)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{pred:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    ax.set_ylabel('Confidence Score', fontsize=12)
                    ax.set_title('Brain Tumor Classification Confidence', fontsize=14, fontweight='bold')
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tumor information
                    if predicted_class != "No Tumor":
                        st.markdown("### 📋 Tumor Information")
                        if predicted_class == "Glioma":
                            st.warning("**Glioma**: Tumors arising from glial cells. Requires immediate medical attention.")
                        elif predicted_class == "Meningioma":
                            st.warning("**Meningioma**: Tumors of the meninges. Usually benign but can cause complications.")
                        elif predicted_class == "Pituitary":
                            st.warning("**Pituitary**: Tumors of the pituitary gland. Can affect hormone production.")
                    else:
                        st.success("✅ **No Tumor Detected**: Great news! The analysis indicates no tumor is present.")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.markdown("""
                    <div style="background-color: #f8d7da; padding: 1rem; border-radius: 10px; border-left: 5px solid #dc3545;">
                        <h5>⚠️ Medical Disclaimer</h5>
                        <p>This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Please try uploading a different image or check the image format.")

if __name__ == "__main__":
    main() 