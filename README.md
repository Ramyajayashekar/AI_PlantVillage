# 🌿 Plant Disease Prediction

## 📌 Overview
This project is a deep learning-based application for predicting plant diseases using image classification. It utilizes a Convolutional Neural Network (CNN) trained on a dataset of plant leaf images to identify various diseases. The model is integrated into a Streamlit web application, allowing users to upload leaf images and receive disease predictions along with recommendations.

## 🚀 Features
- Upload plant leaf images for disease detection.
- AI-based model predicts the disease or pest affecting the plant.
- Provides recommendations based on the identified disease.
- Generates a detailed PDF report of the diagnosis.

## 🔧 Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Web Framework:** Streamlit
- **Image Processing:** OpenCV, PIL
- **Model Storage & Access:** Google Drive (via `gdown`)
- **Report Generation:** ReportLab

## 🛠 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ramyajayashekar/AI_PlantVillage.git
   cd plant-disease-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 📷 How to Use
- Open the Streamlit web app.
- Upload an image of a plant leaf.
- The AI model analyzes the image and predicts the disease (if any).
- View recommendations for disease management.
- Download a detailed Plant Disease Prediction Report (PDF).

## 📊 Model Training
- The model is trained using a labeled dataset of diseased and healthy plant leaves.
- A CNN architecture is used with multiple layers for feature extraction and classification.
- Performance evaluation metrics include accuracy, precision, recall, and F1-score.

## ⚠️ Limitations
- The model's accuracy depends on the quality and variety of training data.
- Might not generalize well to unseen diseases or different environmental conditions.
- Requires periodic retraining with updated datasets for improved accuracy.

## Future Improvements
- Expand the dataset to include more plant species and diseases.
- Implement real-time disease detection using mobile applications.
- Enhance model performance with transfer learning and ensemble methods.

## 📩 Contact
- Email: ramya3152004@gmail.com
- Github: https://github.com/Ramyajayashekar
