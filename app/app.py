import streamlit as st
from PIL import Image
import pandas as pd
from yolo_model.model import model_output
from two_step_model.main import inference_pipeline
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
    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.session_state.uploaded_file = uploaded_file

    model_choice = st.selectbox(
        "Select a model to use:",
        ("YOLOv11", "Two-Step-Model"),
        index=None,  # None = no pre-selection
        placeholder="Choose a model..."
    )

    if uploaded_file is not None and model_choice is not None:
        if model_choice == "YOLOv11":
            model_runner = model_output(webcam=False)
            if st.button("Run"):
                st.info("⏳ Running the selected model, please wait...")
                pil_image = Image.open(uploaded_file)
                orig_img, annotated_img, df = model_runner.run_model(pil_image)
                st.image(annotated_img, caption="Annotated Output", width='stretch')
                st.image(orig_img, caption="Uploaded Image", width='stretch')
                st.dataframe(df)
        elif model_choice == "Two-Step-Model": 
            if st.button("Run"):
                st.info("⏳ Running the selected model, please wait...")
                pil_image = Image.open(uploaded_file)
                results, annotated_img = inference_pipeline(
                    data_folder= 'two_step_model'
                    ,image = pil_image
                    )
                annotated_img = annotated_img[:, :, ::-1]
                st.image(annotated_img, caption="Annotated Output", width='stretch')

elif st.session_state.mode == "webcam":
    pic = st.camera_input("Capture an image")
    st.session_state.pic = pic

    model_choice = st.selectbox(
        "Select a model to use:",
        ("YOLOv11", "Two-Step-Model"),
        index=None,  # None = no pre-selection
        placeholder="Choose a model..."
    )

    if pic is not None and model_choice is not None:
        if model_choice == "YOLOv11":
            model_runner = model_output(webcam=True)
            if st.button("Run"):
                st.info("⏳ Running the selected model, please wait...")
                pil_image = Image.open(pic)
                orig_img, annotated_img, df = model_runner.run_model(pil_image)
                st.image(annotated_img, caption="Annotated Output", width='stretch')
                st.image(orig_img, caption="Captured Image", width='stretch')
                st.dataframe(df)
        elif model_choice == "Two-Step-Model": 
            if st.button("Run"):
                st.info("⏳ Running the selected model, please wait...")
                pil_image = Image.open(pic)
                results, annotated_img = inference_pipeline(
                    data_folder= 'two_step_model'
                    ,image= pil_image
                    )
                annotated_img = annotated_img[:, :, ::-1]
                st.image(annotated_img, caption="Annotated Output", width='stretch')

else:
    st.info("👆 Choose ‘Upload Image’ or ‘Capture Photo’ to get started.")
