import streamlit as st
from PIL import Image
import pandas as pd
from yolo_model.model import model_output
from two_step_model import two_step_pipeline
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

st.markdown("""
<style>
div.stButton > button:first-child {
    background: #00fff2;
    color: white;
    border-radius: 12px;
    font-size: 1.1em;
    font-weight: bold;
    box-shadow: 0 0 20px #1fddff;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 5px #1fddff; }
    50% { box-shadow: 0 0 20px #1fddff; }
    100% { box-shadow: 0 0 5px #1fddff; }
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1fddff, #ff4b1f);
}
</style>
""", unsafe_allow_html=True)


## -- helper functions --
import os
def save_on_disk(uploaded_file):
    """
    Empties the inputs folder, save the uploaded or captured image and returns
    the path

    Args:
        uploaded_file: path in working memory
    
    Returns:
        save_path: path of disk
    """
    folder = "inputs"

    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))
    save_path = os.path.join(folder, f"input_{uploaded_file.type.replace("/", ".")}")

    # Write file to disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path

def app_processing(uploaded_file):
    uploaded_file = save_on_disk(uploaded_file)
    model_choice = st.selectbox(
        "Select a model to use:",
        ("YOLOv11", "Two-Step-Model"),
        index=None,  # None = no pre-selection
        placeholder="Choose a model..."
    )
    if model_choice is not None:
        if model_choice == "YOLOv11":
            model_runner = model_output(webcam=False)
            if st.button("🚀 Run Model"):
                st.info("⏳ Running the selected model, please wait...")
                pil_image = Image.open(uploaded_file)  
                annotated_img, df = model_runner.run_model(pil_image)
                annotated_img.save(uploaded_file.replace(".", "_annotated."))
                df.to_csv(uploaded_file.split(".", 1)[0] + ".csv")
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
            if st.button("🚀 Run Model"):
                st.info("⏳ Running the selected model, please wait...")
                cfg = two_step_pipeline.Config(
                    detector_path = 'two_step_model/BaselineModels/yolo11n-face-best.pt',
                    classifier_path= 'two_step_model/BaselineModels/best_overall.pt'
                )
                det_model = two_step_pipeline.load_detector(cfg)
                cls_model = two_step_pipeline.load_classifier(cfg)
                result = two_step_pipeline.run_on_image(
                    uploaded_file,
                    det_model,
                    cls_model,
                    cfg
                    )
                annotated_img = Image.open(uploaded_file.replace(".", "_annotated."))
                df = two_step_pipeline.faces_to_df(result).drop(columns=['image_path', 'prob_of_emotion'])
                df.to_csv(uploaded_file.split(".", 1)[0] + ".csv")
                if len(df) != 0:
                    st.image(annotated_img, caption="Annotated Output", width='stretch')
                    st.dataframe(df)
                else:
                    st.markdown(f"""
                        <div style='text-align:center; color:#ff4b1f; font-size:20px;'>
                        <b> {no_face_message}
                        </div>
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
        st.subheader("Selected image:")
        st.image(uploaded_file, width='stretch')
        app_processing(uploaded_file)

elif st.session_state.mode == "webcam":
    uploaded_file = st.camera_input("Capture an image")
    st.session_state.uploaded_file = uploaded_file

    if uploaded_file is not None:
        st.subheader("Captured Image:")
        st.image(uploaded_file, width='stretch')
        app_processing(uploaded_file)  
       
else:
    st.info("👆 Choose ‘Upload Image’ or ‘Capture Photo’ to get started.")






st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit & Fine-tuned Vision Models</p>", unsafe_allow_html=True)