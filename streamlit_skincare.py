import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf

# Load the model and data
model = tf.keras.models.load_model('FaceDetectionModel.keras')
csv_df = pd.read_csv('Skincare.csv')

# Page configuration
st.set_page_config(
    page_title="Skincare Recommendation",
    page_icon="💆‍♀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the theme
st.markdown(
    f"""
    <style>
        body {{
            background-color: #feede5;
            font-family: 'Arial', sans-serif;
            color: #333;
        }}
        .stButton>button {{
            background-color: #fea18f;
            color: white;
            border: None;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #e15f44;
            transform: scale(1.05);
        }}
        .stButton>button:active {{
            transform: scale(0.95);
        }}
        .stTitle {{
            font-size: 36px;
            font-weight: bold;
            color: #e15f44;
            text-align: center;
        }}
        .stSubheader {{
            font-size: 24px;
            font-weight: bold;
            color: #fea18f;
        }}
        .stMarkdown {{
            font-size: 16px;
            color: #555;
        }}
        .stImage {{
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }}
        .stTextInput input {{
            border-radius: 15px;
            padding: 10px;
            border: 1px solid #fea18f;
        }}
        .stTextInput input:focus {{
            border-color: #e15f44;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load header image from local directory (remove the upload option)
header_image_path = 'header.png'  # Set the path to your local image
header_image = Image.open(header_image_path)

# Display Header with a beautiful effect
st.image(header_image, use_container_width=True)

# Title and Description with enhanced typography
st.markdown("<h1 class='stTitle'>Find the perfect skincare products for your unique skin type! 💖</h1>", unsafe_allow_html=True)

# Function to prepare the image
def prepare_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Recommendation function
def recommend_skincare(skin_type):
    df = csv_df.copy()
    recommended_products = df[df['skintype'].apply(lambda x: skin_type in x)]
    display_columns = ['product_name', 'brand', 'notable_effects', 'price', 'description', 'product_href', 'picture_src']
    recommended_products = recommended_products[display_columns]
    if recommended_products.empty:
        st.warning("No skincare products found for the selected skin type.")
        return
    recommendations = recommended_products.sample(n=5).reset_index(drop=True)
    st.subheader(f"Top Skincare Recommendations for {skin_type}")
    for idx, row in recommendations.iterrows():
        st.markdown(f"**{row['product_name']} by {row['brand']} - {row['price']}**")
        st.markdown(f"Effects: {row['notable_effects']}")
        st.markdown(f"[Product Link]({row['product_href']})")
        st.image(row['picture_src'], width=150)

# Prediction function
def predict_and_recommend_skin_type(img_path, threshold=65):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)

    # Class labels for skin types
    class_labels = ['Dry', 'Normal', 'Oily', 'Sensitive']
    predicted_probabilities = predictions[0]

    # Create a dictionary of predictions
    prediction_dict = {label: prob for label, prob in zip(class_labels, predicted_probabilities)}

    # Sort predictions by probability
    sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)

    # Determine the highest skin type
    highest_prob_skin_type, highest_prob = sorted_predictions[0]
    st.markdown(f"Predicted Skin Type: **{highest_prob_skin_type}** with {highest_prob * 100:.2f}% probability")

    # Display the image with predicted skin type
    img = load_img(img_path, target_size=(224, 224))
    st.image(img, caption=f"Predicted: {highest_prob_skin_type}", use_container_width=True)

    # Recommend skincare based on the highest skin type
    recommend_skincare(highest_prob_skin_type)

# Camera input for image
camera_image = st.camera_input("Take a photo to analyze your skin type", key="camera_input")

if camera_image:
    # Display the image captured from the camera
    st.image(camera_image, caption="Captured Image", use_container_width=True)

    # Process and predict skin type based on the captured image
    predict_and_recommend_skin_type(camera_image)