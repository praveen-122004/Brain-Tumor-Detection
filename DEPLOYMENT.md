# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account** - You need a GitHub account
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Deployment Steps

### Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click "New repository" or "Create repository"
3. Name it: `brain-tumor-detection`
4. Make it **Public** (required for free Streamlit Cloud)
5. Don't initialize with README (we already have one)

### Step 2: Upload Your Files

**Option A: Using GitHub Desktop**
1. Download and install [GitHub Desktop](https://desktop.github.com/)
2. Clone your repository
3. Copy all project files to the repository folder
4. Commit and push

**Option B: Using Git Commands**
```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Brain Tumor Detection App"

# Add remote repository (replace with your GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-detection.git

# Push to GitHub
git push -u origin main
```

**Option C: Using GitHub Web Interface**
1. Go to your repository on GitHub
2. Click "Add file" → "Upload files"
3. Drag and drop all your project files
4. Commit changes

### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the details:
   - **Repository**: `YOUR_USERNAME/brain-tumor-detection`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `brain-tumor-detection` (or any unique name)
5. Click "Deploy!"

## 📁 Required Files for Deployment

Make sure these files are in your GitHub repository:

```
brain-tumor-detection/
├── app.py                           # Main Streamlit app
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
├── .gitignore                       # Git ignore file
├── models/                          # Model files
│   ├── brain_tumor_cnn_classifier.keras
│   ├── CNN_structure.json
│   └── CNN_weights.pkl
└── mri-images/                      # Sample images (optional)
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## ⚠️ Important Notes

### File Size Limits
- **GitHub**: 100MB per file limit
- **Streamlit Cloud**: 1GB total repository size
- If your model files are too large, consider:
  - Using Git LFS (Large File Storage)
  - Hosting models on cloud storage (Google Drive, AWS S3)
  - Using smaller model formats

### Model File Sizes
- `brain_tumor_cnn_classifier.keras`: ~43MB ✅
- `CNN_weights.pkl`: ~14MB ✅
- `CNN_structure.json`: ~14KB ✅

These should fit within GitHub's limits.

## 🔍 Troubleshooting

### Common Issues:

1. **"App not found" Error**
   - Make sure repository is **public**
   - Check that `app.py` exists in the root directory
   - Verify the main file path is correct

2. **Model Loading Errors**
   - Ensure all model files are uploaded
   - Check file paths in the code
   - Verify model files are not corrupted

3. **Dependency Issues**
   - Check `requirements.txt` has all needed packages
   - Ensure package versions are compatible

4. **Large File Issues**
   - If files are too large, use Git LFS:
   ```bash
   git lfs track "*.keras"
   git lfs track "*.pkl"
   git add .gitattributes
   git add .
   git commit -m "Add large files to LFS"
   git push
   ```

## 🎉 Success!

Once deployed, your app will be available at:
`https://YOUR_APP_NAME-YOUR_USERNAME.streamlit.app`

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are uploaded correctly
3. Test locally first with `streamlit run app.py`
4. Contact Streamlit support if needed

---

**Good luck with your deployment! 🚀** 