import os
import json
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

# Define file paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "../model_training_notebook/model.keras"
class_indices_path = "../model_training_notebook/class_indices.json"

# Load the pre-trained model and class indices
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to Save Image to BytesIO (to handle in-memory image objects)
def save_image_to_buffer(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)  # Reset the pointer to the beginning
    return img_buffer

# Function to Convert the Image to a Temporary File Path
def save_image_to_tempfile(image):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(temp_file, format='JPEG')
    temp_file.close()  # Close the file to ensure it's saved properly
    return temp_file.name  # Return the file path

# Function to Generate a PDF Report with Image Embedded
def generate_pdf_report(prediction, image):
    from reportlab.lib.utils import ImageReader

    # Create a BytesIO object to hold the PDF data
    buffer = BytesIO()

    # Create a canvas for the PDF
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add title and prediction information to the PDF
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Plant Disease Prediction Report")

    # Prediction and recommendations (Shortened Content)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 130, f"Disease/Pest Identified: {prediction}")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 160, "Recommendation:")
    c.drawString(100, height - 190, "- Follow local agricultural guidelines.")
    c.drawString(100, height - 220, "- Consult agricultural experts if necessary.")
    c.drawString(100, height - 250, "This AI-generated report provides insights based on image analysis.")

    # Additional Guidance (Shortened Content)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 280, "Additional Guidance:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 310, "- Inspect crops regularly for early signs of issues.")
    c.drawString(100, height - 340, "- Use treatments recommended by experts.")
    c.drawString(100, height - 370, "- Practice sustainable agricultural methods.")


    # Resize and position the image to appear below the text
    img_width = 300
    img_height = 300
    x_position = (width - img_width) / 2
    y_position = height - 700  # Positioning the image below the text
    image_reader = ImageReader(image)
    c.drawImage(image_reader, x_position, y_position, width=img_width, height=img_height, preserveAspectRatio=True)

    c.save()

    # Get the PDF data from the buffer
    pdf_data = buffer.getvalue()
    buffer.close()

    st.write("""
- **Pepper (Bell) - Bacterial Spot:**  
  Water-soaked spots on leaves and fruits. Treat with copper-based fungicides. Practice crop rotation and avoid overhead irrigation.
  
- **Pepper (Bell) - Healthy:**  
  No issues detected. Continue regular care, ensure optimal watering and nutrient supply.
  
- **Potato - Early Blight:**  
  Dark, concentric spots on older leaves. Treat with fungicides containing chlorothalonil or mancozeb. Remove infected leaves and practice crop rotation.
  
- **Potato - Late Blight:**  
  Irregular water-soaked spots on leaves. Treat with fungicides containing metalaxyl or mefenoxam. Remove infected plants and avoid wet foliage.
  
- **Potato - Healthy:**  
  No issues detected. Continue regular care with balanced fertilization and proper irrigation.
  
- **Tomato - Bacterial Spot:**  
  Small, dark spots with yellow halos on leaves. Use copper-based bactericides. Avoid overhead watering.
  
- **Tomato - Early Blight:**  
  Concentric dark spots on lower leaves. Use fungicides containing chlorothalonil. Remove plant debris.
  
- **Tomato - Late Blight:**  
  Water-soaked, irregular lesions on leaves. Use fungicides containing metalaxyl. Remove affected plants.
  
- **Tomato - Leaf Mold:**  
  Yellow spots on upper leaf surfaces. Treat with copper or sulfur-based fungicides. Ensure good ventilation.
  
- **Tomato - Septoria Leaf Spot:**  
  Small, gray spots with dark borders. Use fungicides with chlorothalonil. Clear plant debris.
  
- **Tomato - Spider Mites (Two-Spotted Spider Mite):**  
  Fine webbing on leaves. Use insecticidal soaps or miticides. Introduce ladybugs.
  
- **Tomato - Target Spot:**  
  Circular, dark brown spots with concentric rings. Use fungicides with azoxystrobin. Remove infected debris.
  
- **Tomato - Tomato Yellow Leaf Curl Virus:**  
  Curling, yellowing leaves. Control whiteflies with insecticidal soaps. Remove infected plants.
  
- **Tomato - Tomato Mosaic Virus:**  
  Mottled leaves with mosaic appearance. Remove infected plants. Practice proper sanitation.
  
- **Tomato - Healthy:**  
  No issues detected. Maintain consistent care with proper watering and fertilization.
""")

    return pdf_data

with st.sidebar:
    st.header("Navigation")
    page = st.selectbox("Go to", ["Home", "Prediction", "About"])

# Home Page
if page == "Home":
    st.title(' 🌿 Welcome to the Plant Disease Classifier')
    st.write("""          
This advanced AI-powered system is designed to assist farmers and agricultural professionals in the early detection and classification of plant diseases. By simply uploading an image of a plant leaf, the system uses deep learning models to quickly analyze the image and provide accurate predictions about potential diseases or pest infestations. The goal is to empower farmers with timely insights, enabling them to take appropriate measures before the issue spreads and causes significant crop damage.

**How It Works**  
The model is trained on a comprehensive dataset of leaf images, covering various plant species and common plant diseases. This ensures accurate predictions that are applicable across a wide range of plants. The system’s user-friendly interface makes it accessible even to individuals with minimal technical expertise, making it an invaluable tool for farmers aiming to enhance crop health and productivity.

**Key Features**  
- **Disease and Pest Detection:**  
  Upload an image of a plant leaf and receive instant classification of potential diseases or pests.  

- **Actionable Recommendations:**  
  Along with predictions, the system provides treatment and prevention strategies to address the detected issues.  

- **User-Friendly Interface:**  
  Designed with simplicity in mind, the platform ensures seamless interaction for users of all skill levels.

**Try It Out**  
Upload an image today to get instant disease classification and actionable recommendations, helping you safeguard your crops and maximize yields.
""")
    st.image("../FarmGaurd.png", use_container_width=True)

# Prediction Page
elif page == "Prediction":
    st.title('🌿 Plant Disease Classifier')
    
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((250, 250))
            st.image(resized_img)

        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

            # Generate PDF report with image included
            pdf_report = generate_pdf_report(prediction, image)

            # Allow user to download the report as a PDF
            download_button = st.download_button(
            label="Download Report",
            data=pdf_report,
            file_name="plant_disease_report.pdf",
            mime="application/pdf"
        )

# About Page
elif page == "About":
    st.title("About the Project")
    st.write("""
             
Project Done By - 
             
1. Ankitha D
2. Dhruthi A R
3. Thanvi B
4. Ramyashree J

**Project Overview**  
This project aims to provide an AI-powered solution for the early detection and classification of plant diseases and pests. Agriculture is a vital industry globally, and crop health is critical to ensuring food security. However, farmers often face challenges in identifying plant diseases or pests during the early stages, leading to substantial crop losses. By leveraging deep learning and computer vision techniques, this system addresses these challenges by offering a fast, accurate, and user-friendly platform for diagnosing plant health issues based on images of leaves.

**How It Works**  
The system utilizes a deep learning model trained on a large and diverse dataset of leaf images. This dataset encompasses various plant species, diseases, and pests, ensuring accurate predictions across different plants. The model employs **Convolutional Neural Networks (CNNs)**, which are well-suited for image classification tasks, to extract meaningful patterns from leaf images and classify them into disease or pest categories. This enables farmers and agricultural professionals to take timely and appropriate actions before the issue affects the entire crop.

**Key Features**  
- **Disease Detection with Actionable Insights:**  
  In addition to identifying diseases and pests, the system provides actionable recommendations such as treatment options and preventive measures. This empowers farmers with the knowledge to address issues effectively.  

- **User-Friendly Interface:**  
  The web interface, built with Streamlit, is designed for simplicity and accessibility, even for users without technical expertise. Farmers can easily upload images of their plant leaves and receive instant results in a seamless and interactive experience.

**Technologies Used**  
- **TensorFlow and Keras:**  
  These libraries were used to build and train the deep learning model for disease classification. Known for their efficiency and widespread adoption in the machine learning community, TensorFlow and Keras are ideal for creating scalable and accurate models.  

- **Streamlit:**  
  The web interface was developed using Streamlit, enabling rapid development of interactive web applications. Its flexibility and simplicity make it perfect for building powerful and user-friendly applications.  

- **Pillow:**  
  This library is utilized for image preprocessing tasks, such as resizing and normalizing the images before they are fed into the deep learning model. Pillow ensures that uploaded images are properly prepared for accurate predictions.

**Significance**  
This project serves as a valuable tool for farmers, agricultural scientists, and anyone interested in plant health. With the increasing global threat of pests and diseases to crops, this technology can play a crucial role in early detection and effective intervention. By integrating AI into agriculture, the system has the potential to reduce crop loss and enhance food security, benefiting farmers and consumers alike.
""")
    
st.write("---")
st.markdown("<p style='text-align: center;'>© 2024 Plant Disease Prediction. All rights reserved.</p>", unsafe_allow_html=True)