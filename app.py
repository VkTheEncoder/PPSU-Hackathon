import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="centered"
)

# --- Title and Description ---
st.title("🩺 Skin Disease Detection AI")
st.write("Upload a photo of the skin condition, and the AI will identify the disease.")

# --- Load the Model ---
# We use @st.cache_resource so the model loads only once, making the app faster
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in the same folder as this file
    model = YOLO("best.pt")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Make sure 'best.pt' is in the folder. Error: {e}")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 2. Add a 'Predict' button
    if st.button('Analyze Image'):
        with st.spinner('Analyzing...'):
            # 3. Run Prediction
            # YOLOv8 can accept a PIL image directly!
            results = model.predict(image)
            
            # 4. Extract Results (Using your logic)
            r = results[0]
            
            # Get the index of the highest probability class
            pred_class_index = r.probs.top1
            
            # Get the confidence score
            confidence = r.probs.top1conf.item() # .item() converts tensor to float
            
            # Get the actual class name
            class_name = r.names[pred_class_index]
            
            # 5. Display Result
            st.success("Analysis Complete!")
            st.metric(label="Predicted Disease", value=class_name)
            st.write(f"**Confidence Score:** {confidence * 100:.2f}%")
            
            # Optional: Warning disclaimer
            st.info("⚠️ Note: This is an AI tool for educational purposes. Always consult a dermatologist for medical advice.")
