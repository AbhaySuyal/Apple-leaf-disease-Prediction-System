import os
import streamlit as st
import tensorflow as tf
import numpy as np

# Function to check if file exists
def file_exists(file_path):
    return os.path.exists(file_path)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model_path = "trained_model11.h5"
    if not file_exists(model_path):
        st.error("Model file 'trained_model.keras' not found. Please ensure the file exists.")
        return None
    
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(360, 360))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) if predictions is not None else None

# Sidebar and main page logic
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("APPLE PLANT LEAF DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Apple Plant Leaf Disease Recognition System! 🌿🔍
        
        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
        
        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.
        
        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
        
        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
        
        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
        This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 14 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        #### Content
        1. train (21,126 images)
        2. test (100+ images)
        3. validation (5,277 images)
        """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True)
    
    if st.button("Predict") and test_image is not None:
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        class_name = ['Black_rot',
                      'Apple___Cedar_apple_rust',
                      'complex',
                      'frog_eye_leaf_spot',
                      'frog_eye_leaf_spot_complex',
                      'healthy',
                      'powdery_mildew',
                      'powdery_mildew_complex',
                      'rust',
                      'rust_complex',
                      'rust_frog_eye_leaf_spot',
                      'scab',
                      'scab_frog_eye_leaf_spot',
                      'scab_frog_eye_leaf_spot_complex']
        
        if result_index is not None:
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        else:
            st.error("Failed to make a prediction. Please try again.")
