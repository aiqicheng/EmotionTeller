import streamlit as st
from PIL import Image
import pandas as pd
from yolo_model.model import model_output
from two_step_model.main import inference_pipeline
from pathlib import Path

st.set_page_config(page_title="Emotion Teller", layout="centered")
#st.title("-- Emotion Teller --")
st.markdown("""
<h1 style='text-align:center;'>
😉 <span style="
background: -webkit-linear-gradient(#ff4b1f, #1fddff);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
">EmoSense</span> 😏
</h1>
""", unsafe_allow_html=True)

no_face_message = "🥲 Oops! Face not detected or emotions couldn't be sensed confidently 😓"

if 'mode' not in st.session_state:
    st.session_state.mode = None

st.markdown("---")
st.subheader("✨ Choose an image to analyze ✨")

#st.write("Select an input source:")

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

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.write("-- Selected image --")
        st.image(pil_image, caption="Uploaded Image", width='stretch')
        
        model_choice = st.selectbox(
            "Select a model to use:",
            ("YOLOv11", "Two-Step-Model"),
            index=None,  # None = no pre-selection
            placeholder="Choose a model..."
        )
        if model_choice is not None:
            if model_choice == "YOLOv11":
                model_runner = model_output(webcam=False)
                if st.button("Run"):
                    st.info("⏳ Running the selected model, please wait...")
                    orig_img, annotated_img, df = model_runner.run_model(pil_image)
                    if len(df) != 0:
                        st.image(annotated_img, caption="Annotated Output", width='stretch')
                        st.dataframe(df)
                    else:
                        st.markdown(f"""
                            <div style='text-align:center; color:#ff4b1f; font-size:20px;'>
                            <b> {no_face_message}
                            </div>
                            """, unsafe_allow_html=True)
            elif model_choice == "Two-Step-Model": 
                if st.button("Run"):
                    st.info("⏳ Running the selected model, please wait...")
                    df, annotated_img = inference_pipeline(
                        data_folder= 'two_step_model'
                        ,image= pil_image
                        )
                    if len(df) != 0:
                        annotated_img = annotated_img[:, :, ::-1]
                        st.image(annotated_img, caption="Annotated Output", width='stretch')
                        st.dataframe(df)
                    else:
                        st.markdown(f"""
                            <div style='text-align:center; color:#ff4b1f; font-size:20px;'>
                            <b> {no_face_message}
                            </div>
                            """, unsafe_allow_html=True)

elif st.session_state.mode == "webcam":
    pic = st.camera_input("Capture an image")
    st.session_state.pic = pic

    if pic is not None:
        pil_image = Image.open(pic)
        st.write("-- Selected image --")
        st.image(pil_image, caption="Uploaded Image", width='stretch')
    
        model_choice = st.selectbox(
            "Select a model to use:",
            ("YOLOv11", "Two-Step-Model"),
            index=None,  # None = no pre-selection
            placeholder="Choose a model..."
        )
        if model_choice is not None:
            if model_choice == "YOLOv11":
                model_runner = model_output(webcam=True)
                if st.button("Run"):
                    st.info("⏳ Running the selected model, please wait...")
                    orig_img, annotated_img, df = model_runner.run_model(pil_image)
                    if len(df) != 0:
                        st.image(annotated_img, caption="Annotated Output", width='stretch')
                        st.dataframe(df)
                    else:
                        st.markdown(f"""
                            <div style='text-align:center; color:#ff4b1f; font-size:20px;'>
                            <b> {no_face_message}
                            </div>
                            """, unsafe_allow_html=True)
            elif model_choice == "Two-Step-Model": 
                if st.button("Run"):
                    st.info("⏳ Running the selected model, please wait...")
                    df, annotated_img = inference_pipeline(
                        data_folder= 'two_step_model'
                        ,image= pil_image
                        )
                    if len(df) != 0:
                        annotated_img = annotated_img[:, :, ::-1]
                        st.image(annotated_img, caption="Annotated Output", width='stretch')
                        st.dataframe(df)
                    else:
                        st.markdown(f"""
                            <div style='text-align:center; color:#ff4b1f; font-size:20px;'>
                            <b> {no_face_message}
                            </div>
                            """, unsafe_allow_html=True)
else:
    st.info("👆 Choose ‘Upload Image’ or ‘Capture Photo’ to get started.")

with st.expander("ℹ️ How it works"):
    st.write("This app uses fine-tuned vision models to classify facial expressions into emotions. Just click or upload an image, choose a model from the dropdown, hit run, and watch the magic happen")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit & Fine-tuned Vision Models</p>", unsafe_allow_html=True)
