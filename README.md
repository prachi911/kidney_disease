# 🧠 Kidney Disease Classifier | Course Project CAP-306

A deep-learning-based medical image classifier application classifying Computed Tomography images into one of 4 categories(Health/Normal, Cyst, Tumor, or Stone). The application aims to assist doctors with diagnoses by identifying features and details that might get overlooked, acting as a second opinion. The application also provides a number of other features to assist in treatment and recovery. 

---

## The numbers: 
input shape: (224, 224, 3)  
Final accuracy: 98.84%   
Trainable Parameters: 488,580   
Non-trainable Parameters: 960   

---

## 🩺 Features
- 📷 Upload kidney scan images (CT)
- Global max-pooling for performance
- 🧠 Custom CNN-based classification:  
  - Normal  
  - Cyst  
  - Tumor  
  - Stone
- 🧾 Recovery time estimation
- 📊 Dynamic health insights and recommendations
- 💡 Clean, user-friendly Streamlit interface

---

## 🚀 Tech Stack
- **Language**: Python(3.9 recommended) 
- **Frontend & backend**: StreamLit
- **Model**: CNN (Convolutional Neural Network)
- **Libraries**: TensorFlow / Keras, OpenCV, NumPy, Pillow

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/kidney-disease-classifier.git
cd kidney-disease-classifier
```
### 2. Create virtual environment (optional but recommended)
using venv: 
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```
using conda:
```bash
conda create -n env python=3.9
conda activate env
```
### 3. install req
```bash
pip install -r requirements.txt
```
### 4.Lauch the app
```bash
streamlit run app.py
```

---

## co-contributors: 
- Aditya Pratap Singh
- Shruti Tiwari


