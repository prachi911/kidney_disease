# 🧠 Kidney Disease Classifier

A deep learning-powered Streamlit app that classifies kidney conditions (Normal, Cyst, Tumor, Stone) from medical images. The app also provides estimated recovery time and smart insights based on predictions to assist in early detection and awareness.

> ⚕️ Your AI assistant for preliminary kidney diagnosis.

---

## 🩺 Features

- 📷 Upload kidney scan images (CT/MRI)
- 🧠 CNN-based classification:  
  - Normal  
  - Cyst  
  - Tumor  
  - Stone
- 🧾 Recovery time estimation
- 📊 Dynamic health insights and recommendations
- 💡 Clean, user-friendly Streamlit interface

---

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Model**: CNN (Convolutional Neural Network)
- **Libraries**: TensorFlow / Keras, OpenCV, NumPy, Pillow

---

## 📸 Preview

![App Screenshot](assets/kidney_app_preview.png)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/kidney-disease-classifier.git
cd kidney-disease-classifier
```
### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```
### 3. install req
```bash
pip install -r requirements.txt
```
### 4.Lauch the app
```bash
streamlit run app.py
```

