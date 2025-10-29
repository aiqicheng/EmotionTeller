import streamlit as st
from PIL import Image
import pandas as pd
from model import model_output
from pathlib import Path

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("-- Emotion Detection Demo --")

if 'mode' not in st.session_state:
    st.session_state.mode = None

st.write("Select an input source:")

col1, col2 = st.columns(2)

with col1:
    if st.button("📁 Upload Image"):
        st.session_state.mode = "upload"

with col2:
    if st.button("📸 Capture Photo"):
        st.session_state.mode = "webcam"

st.write(f"**Mode selected:** `{st.session_state.mode}`")

if st.session_state.mode == "upload":
    model_runner = model_output(upload=True, webcam=False)

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    st.session_state.uploaded_file = uploaded_file

    if uploaded_file is not None:
        if st.button("Run Model"):
            st.write("Wait for your result...")
            pil_image = Image.open(uploaded_file)
            orig_img, annotated_img, df = model_runner.run_model(image=pil_image)
            st.image(annotated_img, caption="Annotated Output", width='stretch')
            st.image(orig_img, caption="Original Image", width='stretch')
            st.dataframe(df)

elif st.session_state.mode == "webcam":
    model_runner = model_output(upload=False, webcam=True)

    pic = st.camera_input("Capture an image")
    st.session_state.pic = pic

    if pic is not None:
        if st.button("Run Model"):
            st.write("Wait for your result...")
            pil_image = Image.open(pic)
            orig_img, annotated_img, df = model_runner.run_model(pil_image)
            st.image(annotated_img, caption="Annotated Output", width='stretch')
            st.image(orig_img, caption="Captured Image", width='stretch')
            st.dataframe(df)

else:
    st.info("👆 Choose ‘Upload Image’ or ‘Capture Photo’ to get started.")
