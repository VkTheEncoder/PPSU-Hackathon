import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# --- CONFIGURATION ---
# PASTE YOUR GOOGLE API KEY HERE
GOOGLE_API_KEY = "AIzaSyAafyKfEWI30jEolmxBvWUo9qQe58cVhzU"

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Page Configuration ---
st.set_page_config(
    page_title="SkinCare AI (Gemini Powered)",
    page_icon="🩺",
    layout="centered"
)

# --- CSS Hack for Clean Look ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "detected_disease" not in st.session_state:
    st.session_state.detected_disease = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# --- Load YOLO ---
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

def get_ai_response(user_question, image, disease_context):
    try:
        # Use Gemini 1.5 Flash (Fast & Multimodal)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # We give the AI:
        # 1. The Context (What YOLO found)
        # 2. The User's Question
        # 3. The ACTUAL IMAGE (Crucial Upgrade!)
        
        prompt = f"""
        You are an expert Dermatologist AI. 
        Analysis Context: A separate YOLO model detected this as '{disease_context}'.
        
        User Question: {user_question}
        
        Task: 
        1. Look at the image provided to confirm if the visual symptoms match the detection.
        2. Answer the user's question professionally.
        3. If the image looks totally different from '{disease_context}', politely mention that.
        4. Keep it concise and helpful.
        """
        
        # Send text + image to Gemini
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error connecting to Google Gemini: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🩺 Skin Scanner")
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and st.button('🚀 Analyze Condition', type="primary"):
        with st.spinner('Scanning image...'):
            # 1. Image Setup
            image = Image.open(uploaded_file)
            st.session_state.current_image = image # Save image to memory for the chat
            
            # 2. Run YOLO (Fast Initial Check)
            model = load_yolo_model()
            results = model.predict(image)
            r = results[0]
            disease_name = r.names[r.probs.top1]
            confidence = r.probs.top1conf.item()
            
            # 3. Save Context
            st.session_state.detected_disease = disease_name
            
            # 4. Greeting
            analysis_msg = (
                f"### 🔬 Analysis Result\n"
                f"**YOLO Detection:** {disease_name}\n"
                f"**Confidence:** {confidence*100:.1f}%\n\n"
                f"I have analyzed the image pixels. You can now ask me detailed questions."
            )
            
            st.session_state.messages = [{"role": "assistant", "content": analysis_msg}]
            st.rerun()

    st.divider()
    if st.session_state.detected_disease:
        st.success(f"Target: **{st.session_state.detected_disease}**")

# --- MAIN CHAT ---
st.title("SkinCare AI Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the condition..."):
    if st.session_state.current_image:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing image and answering..."):
                # Send the IMAGE + QUESTION to Gemini
                response = get_ai_response(prompt, st.session_state.current_image, st.session_state.detected_disease)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload an image first!")
