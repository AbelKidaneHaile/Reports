import streamlit as st
from PIL import Image
from prediction import prediction
from prediction import confidence
from prediction import iou_thresold
from prediction import Display_Confidence
from prediction import Display_Class
import streamlit as st
import time
import os

# Global variables
uploaded_file = None
path_to_image = ""

def make_prediction():

    global confidence
    global path_to_image
    global uploaded_file
    global iou_thresold

    if uploaded_file is not None:
        with st.spinner(f"Detecting heads in the image: {uploaded_file.name}"):
            annotatedImage = prediction(path_to_image, confidence, 
            disp_Class=Display_Class, disp_Confidence=Display_Confidence)
        st.image(annotatedImage, caption='Model Prediction')
    
def upload_file():

    global path_to_image
    global uploaded_file
    global confidence

    uploaded_file = st.file_uploader("Upload an image",type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        path_to_image = "image/"+uploaded_file.name
        image = Image.open(uploaded_file)
        # st.image(image, caption="Original image")
        # Save image to the directory 'image' if it doesn't exist
        if not os.path.exists(path_to_image):
            image.save(path_to_image)
        make_prediction() # make prediction


def side_bar():

    global confidence
    global uploaded_file
    global iou_thresold
    global Display_Confidence
    global Display_Class

    with st.sidebar:
        st.subheader("Modify parameters")
        confidence = st.slider('Confidence %', 0, 100, 80)
        iou_thresold = st.slider('IOU Threshold %', 0, 100, 30)
        
        # Checkboxes to display class and confidence for each detection
        Display_Class = st.checkbox('Display Class', value=True)             #  >> In Development
        Display_Confidence = st.checkbox('Display Confidence', value=True) 

        if uploaded_file is not None:
            make_prediction() # make prediction
            st.text(f'{Display_Class} {Display_Confidence}')



def main_func():
    #Title
    st.title('YoloV8 Head Detector')
    #description
    st.text('This is a YoloV8 object detection model that detects human heads.')
    
    side_bar()
    upload_file()



if __name__=='__main__':
    main_func()