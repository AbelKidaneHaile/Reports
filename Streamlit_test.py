import streamlit as st
from PIL import Image
from prediction import prediction
import streamlit as st

#Title
#description
#Sidebar that contains parameters that can display confidence, and other parameters
#




image_path = st.text_input('Image path', 'Enter the path to the image')
st.write('Image path is', image_path)

if st.button('View Image'):
    image = Image.open(image_path)
    st.image(image, caption='Original Image')



model_path = st.text_input('Model path', 'Enter the path to the model')
st.write('Model path is', model_path)

if st.button('View Prediction'):
    annotatedImage = prediction(image_path, model_path)
    st.image(annotatedImage, caption='Model Prediction')
