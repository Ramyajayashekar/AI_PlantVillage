🌿 Plant Disease Prediction
A deep learning-based application for detecting plant diseases from leaf images using a Streamlit web interface.

📌 Overview
This project leverages deep learning techniques to identify plant diseases from leaf images. The model is trained on a dataset of various plant diseases and healthy leaves. A user-friendly Streamlit application allows users to upload images and receive predictions along with recommendations for disease management.

🚀 Features
Upload plant leaf images for disease prediction
AI-powered classification using deep learning (CNN model)
Provides disease name and treatment recommendations
Generates a downloadable Plant Disease Prediction Report (PDF)
Easy-to-use Streamlit web interface

🔧 Technologies Used
Deep Learning: TensorFlow, Keras
Web Framework: Streamlit
Model Training & Evaluation: Python, OpenCV, NumPy, Pandas
Data Handling: Google Drive (for model storage), gdown
Report Generation: ReportLab

🛠 Installation & Setup
Clone the repository:
bash
Copy
Edit
git clone https://github.com/Ramyajayashekar/AI_PlantVillage.git
cd plant-disease-prediction
Create a virtual environment and activate it:
bash
Copy
Edit
python -m venv venv  
source venv/bin/activate  # For MacOS/Linux  
venv\Scripts\activate  # For Windows  
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt  
Run the Streamlit application:
bash
Copy
Edit
streamlit run app/main.py  
📷 How to Use
Open the Streamlit web app.
Upload an image of a plant leaf.
The AI model analyzes the image and predicts the disease (if any).
View recommendations for disease management.
Download a detailed Plant Disease Prediction Report (PDF).
