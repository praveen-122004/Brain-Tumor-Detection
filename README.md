# 🧠 Brain Tumor Detection System

An AI-powered web application for detecting brain tumors from MRI scans using Convolutional Neural Networks (CNN).

## 🎯 Features

- **Real-time Analysis**: Upload brain MRI scans and get instant predictions
- **Multi-class Classification**: Detects 4 types of brain conditions:
  - Glioma
  - Meningioma  
  - Pituitary Tumor
  - No Tumor (Normal)
- **Interactive Interface**: User-friendly Streamlit web interface
- **Detailed Results**: Confidence scores and tumor information
- **Medical Information**: Educational content about each tumor type

## 🛠️ Installation

1. **Clone or download this project**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your model files are in the correct location**:
   ```
   models/
   ├── brain_tumor_cnn_classifier.keras
   ├── CNN_structure.json
   └── CNN_weights.pkl
   ```

## 🚀 Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Use the application**:
   - Navigate to "Tumor Detection" page
   - Upload a brain MRI image (JPG, PNG, JPEG)
   - Click "Analyze Image" to get results

## 📁 Project Structure

```
Brain tumor/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── load_model_get_weights.py      # Model loading utility
├── tumor-classification-cnn.ipynb  # Training notebook
├── models/                         # Trained model files
│   ├── brain_tumor_cnn_classifier.keras
│   ├── CNN_structure.json
│   └── CNN_weights.pkl
└── mri-images/                     # Sample MRI images
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 🔬 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224×224×3 pixels
- **Output**: 4-class classification with softmax activation
- **Training**: Categorical crossentropy loss with Adamax optimizer

## 📊 Usage Instructions

### Home Page
- Overview of the system
- Quick start guide
- Model statistics

### Tumor Detection Page
1. **Upload Image**: Select a brain MRI scan file
2. **View Image Info**: Check file details and image properties
3. **Analyze**: Click the analyze button to process the image
4. **Review Results**: 
   - Predicted tumor type
   - Confidence scores
   - Detailed tumor information
   - Medical symptoms and treatment options

### About Page
- Technical details about the model
- Information about different tumor types
- Technology stack used

### Help Page
- Step-by-step usage instructions
- Troubleshooting guide
- Image requirements and limitations

## ⚠️ Important Disclaimers

**Medical Disclaimer**: 
- This tool is for **educational and research purposes only**
- **NOT a substitute for professional medical diagnosis**
- Always consult qualified healthcare professionals for medical decisions
- Results should be interpreted by medical personnel

**Accuracy**: 
- Model accuracy is high but not 100%
- False positives and negatives are possible
- Clinical validation is required for medical use

## 🎨 Features

### Interactive Interface
- Modern, responsive design
- Real-time image processing
- Visual confidence plots
- Detailed tumor information

### Image Processing
- Automatic resizing to 224×224 pixels
- RGB channel normalization
- Support for various image formats
- Grayscale to RGB conversion

### Results Visualization
- Bar charts showing prediction confidence
- Color-coded tumor type indicators
- Detailed medical information for each tumor type

## 🔧 Technical Requirements

- Python 3.7+
- TensorFlow 2.13.0
- Streamlit 1.28.0
- Pillow 10.0.0
- Other dependencies listed in `requirements.txt`

## 🐛 Troubleshooting

### Common Issues

**Model Loading Error**:
- Ensure model files exist in the `models/` directory
- Check file permissions
- Verify TensorFlow version compatibility

**Image Upload Issues**:
- Use supported formats: JPG, PNG, JPEG
- Ensure file size is under 10 MB
- Check image quality and resolution

**Slow Processing**:
- Large images may take longer to process
- Check system resources
- Ensure stable internet connection

## 📈 Performance

- **Processing Time**: < 5 seconds per image
- **Model Accuracy**: High precision classification
- **Supported Formats**: JPG, PNG, JPEG
- **File Size Limit**: 10 MB maximum

## 🤝 Contributing

This is an educational project. For improvements or suggestions:
1. Review the code structure
2. Test with different image types
3. Consider adding new features or improvements

## 📄 License

This project is for educational purposes. Please ensure compliance with any applicable licenses for the training data and models used.

---

**Note**: This application is designed for educational and research purposes. For medical diagnosis, always consult qualified healthcare professionals. 