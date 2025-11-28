import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------------------

# Streamlit UI

# ---------------------------

st.set_page_config(page_title="😄 Facial Expression Detection", layout="wide")
st.title("😄 Facial Expression Detection (VGG16 Model)")
st.sidebar.title("Options")
mode = st.sidebar.radio("Select Mode:", ["Upload Image", "Live Webcam"])


# ---------------------------

# Load Model

# ---------------------------

@st.cache_resource
def load_expression_model():
    model = tf.keras.models.load_model("best_vgg16_model.h5")
    return model

model = load_expression_model()

# Class names

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_EMOJIS = ["😠", "🤢", "😨", "😄", "😐", "😢", "😲"]

# ---------------------------

# Preprocessing Function

# ---------------------------

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    norm = resized.astype("float32") / 255.0
    final_image = np.expand_dims(norm, axis=(0, -1))
    final_image = np.repeat(final_image, 3, axis=-1)  # (1,48,48,3)
    return final_image

#---------------------------
#Webcam Transformer
#--------------------------

class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = preprocess_frame(img)
        predictions = model.predict(processed)
        class_index = np.argmax(predictions)
        label = CLASS_NAMES[class_index]
        emoji = CLASS_EMOJIS[class_index]
        text = f"{label.upper()} {emoji}"
        # Overlay text on frame
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
        return img


# ---------------------------

# Upload Image Mode

# ---------------------------

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(img, channels="BGR", caption="Uploaded Image", width=300)
        processed = preprocess_frame(img)
        predictions = model.predict(processed)
        class_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[class_index]
        predicted_emoji = CLASS_EMOJIS[class_index]

        st.subheader("Prediction:")
        st.write(f"**{predicted_label.upper()} {predicted_emoji}**")
        st.write("Raw probabilities:", predictions)

# ---------------------------

# Live Webcam Mode

# ---------------------------

elif mode == "Live Webcam":
    st.write("Webcam Live Detection:")

webrtc_streamer(key="emotion", video_transformer_factory=EmotionTransformer)

