from ultralytics import YOLO
import time
import streamlit as st
import cv2 #module to work with images
import numpy as np
import tempfile #module for creating temporary files and directories
import os

def load_model(model_path):
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml")) #bytetrack-for real-time multi-obj detection by creating unique ID for each obj displaying results of multiple objs on a single frmae, #reidentification module-one image detection in 1 frame
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    
    # Resize the image to a standard size
    image = cv2.resize(image, (640, int(640*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def video(conf, model):
    is_display_tracker, tracker = display_tracker_options()
    col1, col2 = st.columns(2)
    VIDEO_1_PATH = r"D:\VAISH 360 ASSIGNMENTS\CRACK PROJ INNO\model deployment\VID_20231128_162243.mp4"
    VIDEOS_DICT = {
        'video_1': VIDEO_1_PATH}

    source_vid = st.selectbox(
        "Choose a video...", VIDEOS_DICT.keys())
    with col1:
        try:
            if source_vid is None:
                st.write('PLEASE UPLOAD THE VIDEO')

            else:
                with open(VIDEOS_DICT.get(source_vid), 'rb') as video_file:
                    video_bytes = video_file.read()
                if video_bytes:
                    st.video(video_bytes)
    # Videos config
        except Exception as ex:
            st.error("Error occurred while opening the Video.")
            st.error(ex)

    with col2:
        if st.button('Detect Video Objects'):
            try:
                vid_cap = cv2.VideoCapture(
                    str(VIDEOS_DICT.get(source_vid)))
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf,
                                                 model,
                                                 st_frame,
                                                 image,
                                                 is_display_tracker,
                                                 tracker
                                                 )
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.error("Error loading video: " + str(e))