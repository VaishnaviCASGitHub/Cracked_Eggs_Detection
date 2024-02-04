# -*- coding: utf-8 -*-

#pip install torch- ML training framework for deep learning

#pip install ultralytics- for ML object detection 

#pip install streamlit- app framework for ML and DS

from pathlib import Path #for path handling 
import PIL #python imaging library 
import helper #helper file
import streamlit as st
import time #time library
# Creating main page heading
html_temp = """
     <!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>Vaishnavi Chennupalli || Egg-crack-detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
h1{
	font-family: 'Raleway', sans-serif;
	font-weight: 500;
        color: aqua;
        padding: 10px 0px;
}
h3{
	font-family: 'Raleway', sans-serif;
	font-weight: 100;
        color: #86CDEE;
        padding: 10px 0px;
}
</style>
<body>
<h1>EGG CRACK DETECTION</h1>
<h3>A YOLOv8 Project at INNODATATICS</h3>
<a href="https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/"  target="_blank">For more details on YOLOv8 visit this site!</a>
</body>
</html>
"""

st.markdown(html_temp, unsafe_allow_html = True)
with st.sidebar:
     radio_side=st.sidebar.radio("MENU", ("Project details", "About Us", "Contact"))
if radio_side == 'Project details':
  st.sidebar.write("Click here [link](https://drive.google.com/drive/folders/1QbFUBUB7IiLxd1FThr1fMCPlvw5XFokB?usp=sharing)")
if radio_side == 'About Us':
  st.sidebar.write("Click here [link](https://innodatatics.ai/business_solutions)")
if radio_side == 'Contact':
  st.sidebar.write("Click here [link](https://innodatatics.ai/contact)")
#Main part 
#Selecting Detection Or Segmentation
model_path = r"D:\VAISH 360 ASSIGNMENTS\CRACK PROJ INNO\runs\detect\train\weights\best.pt"

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load the model. Check the specified path: {model_path}")
    st.error(ex)
html_temp = """
     <!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>Vaishnavi Chennupalli || Egg-crack-detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
h4{
	font-family: 'Raleway', sans-serif;
	font-weight: 500;
        color: white;
        padding: 10px 0px;
}
</style>
<body>
<br> </br>
<h4>Egg crack images detection</h4>
</body>
</html>
"""

st.markdown(html_temp, unsafe_allow_html = True)

# Adding file uploader for selecting images
from PIL import Image
image = Image.open(r"C:\Users\C Adinarayana\Pictures\Screenshots\Screenshot (2394).png")
st.image(image, caption='Sample model image',
                     use_column_width=True)
# Selecting Detection Or Segmentation
model_path = r"D:\VAISH 360 ASSIGNMENTS\CRACK PROJ INNO\runs\detect\train\weights\best.pt"

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load the model. Check the specified path: {model_path}")
    st.error(ex)

html_temp = """
     <!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>Vaishnavi Chennupalli || Egg-crack-detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
h5{
	font-family: 'Raleway', sans-serif;
	font-weight: 500;
        color: teal;
        padding: 10px 0px;
}
</style>
<body>
<br> </br>
<h5>IMAGE & VIDEO DETECTION</h5>
</body>
</html>
"""

st.markdown(html_temp, unsafe_allow_html = True)
IMAGE = 'Image'
VIDEO = 'Video'
SOURCES = [IMAGE, VIDEO]
source_radio = st.radio(
    "Select Source", SOURCES)
confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
source_img = None
# If image is selected
if source_radio == 'Image':
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp", "mov"))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                st.write("PLEASE UPLOAD THE IMAGE")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Unknown error has occurred while opening the image.")
            st.error(ex)

    with col2:
        if st.button('Detect Objects'):
            with st.spinner('Wait for it...'):
                 time.sleep(5)
                 st.success('Your results!')
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No image is uploaded yet!")
elif source_radio == 'Video':
    helper.video(confidence, model)
else:
    st.error("Kindly select a valid source!")
html_temp = """
     <!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>Vaishnavi Chennupalli || Egg-crack-detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
h6{
	font-family: 'Raleway', sans-serif;
	font-weight: 10;
        color: #3b3b3b;
        padding: 50px 300px;
}
</style>
<body>
<br> </br>
<h6 use_column_width=True>Made by VaishnaviCAS</h6>
</body>
</html>
"""
st.markdown(html_temp, unsafe_allow_html = True)
html_temp = """
<!DOCTYPE html>
<html lang="en">
<meta charset="UTF-8">
<title>Vaishnavi Chennupalli || Egg-crack-detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<!-- <link rel="stylesheet" href="C:/Users/C Adinarayana/Downloads/style.css"> -->
<style>
.text-white{
            color: white;
}
.bottomnav
{
            text-align: center;
            padding: 0px 150px;
            text-decoration: none;
            font-size: 17px;
}
.bottomnav a
{
           float: right;
           color: white;
           text-align: left;
           padding: 0px 150px;
           text-decoration: none;
           font-size: 17px;
}
.bottomnav a:hover
{
           background-color: #ddd;
           color: #F39F5A;
}
.bottomnav a:active
{
           background-color: #04AA6D;
           color: white;
}
</style>
<div class="bottomnav">
          <a class="text-white" href="https://www.linkedin.com/in/vaishnavi-c-a-s-chennupalli" target="_blank">Linkedin</a>
          <a class="text-white" href="https://github.com/VaishnaviCASGitHub"  target="_blank">Github</a>
</div>
</body>
</html>
"""
st.markdown(html_temp, unsafe_allow_html = True)

