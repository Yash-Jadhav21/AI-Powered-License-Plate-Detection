import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pytesseract
import easyocr
import re
from collections import Counter
from datetime import datetime
import subprocess
import tempfile
os.environ['EASYOCR_MODULE_PATH'] = tempfile.gettempdir()

# OCR setup
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)


# UI setup
st.set_page_config(page_title="YOLO License Plate Detector", layout="centered", page_icon="🚘")
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        h1 {color: #0b3d91;}
        .stButton>button {
            background-color: #0099ff; 
            color: white; 
            font-weight: bold; 
            border-radius: 10px;
        }
        .stFileUploader {
            border: 2px dashed #0099ff; 
            border-radius: 10px; 
            padding: 10px;
        }
        .css-1cpxqw2 {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI-Powered License Plate Recognition System")

# Folder setup
if not os.path.exists("temp"):
    os.makedirs("temp")

if st.button("🔁 Reset"):
    st.rerun()

uploaded_file = st.file_uploader(
    "📤 Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# Load YOLO model
try:
    model = YOLO('C:\\Users\\yj428\\OneDrive\\Desktop\\ALPD\\best.pt')  # Update path if needed
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

# Helper functions
def clean_plate_text(text):
    text = text.upper().replace(" ", "")
    text = text.replace('I', '1').replace('O', '0').replace('Q', '0')
    return re.sub(r'[^A-Z0-9]', '', text).strip()

def extract_text_from_image(image):
    try:
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(image)
        if results:
            results = sorted(results, key=lambda x: -x[2])
            return clean_plate_text(results[0][1])
        return ""
    except Exception as e:
        st.error(f"EasyOCR error: {e}")
        return ""

# Image prediction
def predict_and_save_image(path_test_car, output_image_path):
    results = model.predict(path_test_car, device='cpu')
    image = cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    extracted_texts = []

    for result in results:
        boxes = sorted(result.boxes, key=lambda b: b.conf[0], reverse=True)
        if boxes:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f'{box.conf[0]*100:.2f}%', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            plate_img = image[y1:y2, x1:x2]
            text = extract_text_from_image(plate_img)
            if text: extracted_texts.append(text)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, image)
    return output_image_path, extracted_texts

# Video prediction (most frequent plate only)
def predict_and_plot_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video.")
        return None, []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = "temp/temp_raw_output.mp4"
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

    all_texts = []
    frame_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, device='cpu')

        for result in results:
            for box in result.boxes:
                frame_boxes.append((frame.copy(), box))

    cap.release()

    for frame, box in frame_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_img = frame[y1:y2, x1:x2]
        text = extract_text_from_image(plate_img)
        if text: all_texts.append(text)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0),2)
        cv2.putText(frame, f'{box.conf[0]*100:.2f}%', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
        out.write(frame)

    out.release()

    ffmpeg_path = r'C:\Users\\yj428\\Downloads\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe'
    subprocess.run([ffmpeg_path,'-y','-i',temp_output,'-vcodec','libx264','-acodec','aac',output_path], check=False)

    if all_texts:
        most_common_text, _ = Counter(all_texts).most_common(1)[0]
        return output_path, [most_common_text]
    else:
        return output_path, []

# Unified handler
def process_media(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    else:
        return predict_and_save_image(input_path, output_path)

# Handle upload
if uploaded_file:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info("📽️ Processing media...")

    result_path, texts = process_media(input_path, output_path)

    # Display processed output
    if result_path.endswith(('.mp4','avi','mov','mkv')):
        st.video(result_path)
    else:
        st.image(result_path)

    # Show extracted text and details
    if texts:
        for plate in texts:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"✅ Extracted License Plate: `{plate}`")
            st.info(f"🕒 Timestamp: {timestamp}")
    else:
        st.warning("⚠️ No license plate detected.")

