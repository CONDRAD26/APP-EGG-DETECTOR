import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar 
st.sidebar.markdown(
        """
        <div style="text-align: ;">
            <h1 style="color: #ff6347;">kuevos AI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox("Select page",["Home", "About", "Egg Recognition","Egg Recognition(take a photo)"])

# Home page
if(app_mode == "Home"):
    st.header("Egg Class Recognition System")
    image_path ="image.jpeg"
    st.image(image_path, width=450) 
    st.markdown("""
## Welcome to Our Egg Class Detection App.

### About Us

Welcome to the Egg Class Detection App! Our goal is to provide a simple and efficient tool for classifying different types of eggs for customer to get the class of Eggs the want . This application is only for internal use only no any other individuals or organizations should use without permission.

### Our Mission

Our mission is to empower the egg industry with advanced technology, enhancing productivity and accuracy.

### How It Works

1. **Upload an image of the egg.**
2. **Our app processes the image using a powerful machine learning model.**
3. **Instantly receive the classification result.**

### Get Started

Simply upload an egg image (JPG, JPEG, PNG) and let our app handle the rest. Thank you for choosing our Egg Class Detection App!
""")
    
    
#About us
elif(app_mode =="About"):
    st.header("About ")
    st.markdown("""
    ### About Dataset
    The dataset was collected on a farm and comprises images of eggs, categorized into four distinct classes: Class AA, Class A, Class B, and Stale. This dataset is crucial for developing a machine learning model aimed at classifying eggs based on their quality. The images were captured in a controlled environment to ensure consistency in lighting and background, providing a reliable basis for training and testing the model. Each class represents a specific quality of eggs, with clear visual distinctions that facilitate accurate classification.
    """) 
            
#Prediction Page 
elif(app_mode == "Egg Recognition"):
    st.header("Egg Recognition")
    test_image = st.file_uploader("Choose an Image")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
        #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait....."):
            st.write("Our prediction")
            result_index = model_prediction(test_image)
        #Define Classname
            class_name = ['Egg_Class B', 'Egg_Fresh Class AA', 'Egg_Fresh Class B', 'Egg_stale']
            st.success("model is Predicting it's a {}".format(class_name[result_index]))
        
#Egg Recognition2
elif(app_mode == "Egg Recognition(take a photo)"):
    st.header("Egg Recognition(take a photo)")
    test_image = st.camera_input("Take an Image")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
        #Predict Button
    if(st.button("Predict")):
        with st.spinner("Please Wait....."):
            st.snow()
            st.write("Our prediction")
            result_index = model_prediction(test_image)
        #Define Classname
            class_name = ['Egg_Class B', 'Egg_Fresh Class AA', 'Egg_Fresh Class B', 'Egg_stale']
            st.success("model is Predicting it's a {}".format(class_name[result_index]))