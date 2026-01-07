import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# --- CONFIGURATION ---
# PASTE YOUR GOOGLE API KEY HERE
GOOGLE_API_KEY = "AIzaSyCESMVO_w2ZtYjOJz8elB0e-U8pVFIszsg"

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Page Configuration ---
st.set_page_config(
    page_title="SkinCare AI Assistant",
    page_icon="🩺",
    layout="centered"
)

# --- CSS for Clean Chat UI ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "detected_disease" not in st.session_state:
    st.session_state.detected_disease = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# --- Load YOLO Model ---
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

# --- AI FUNCTIONS ---

def generate_initial_report(image, disease_context):
    """
    Generates the FIRST detailed report immediately after analysis.
    """
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        You are an expert Dermatologist.
        
        Task: Create a structured initial medical report for the patient.
        
        Inputs:
        1. User's Image (Attached)
        2. AI Detection Hint: "{disease_context}" (This comes from a YOLO model, verify it visually).
        
        Output Format (Use Markdown):
        ## 🩺 Diagnosis Report
        
        **Visual Assessment:** [Confirm if the image matches "{disease_context}" or looks like something else.]
        
        ### 📖 What is it?
        [Brief explanation of the condition]
        
        ### ⚠️ Common Symptoms
        * [Symptom 1]
        * [Symptom 2]
        
        ### 💊 Suggested Treatments & Care
        * [Treatment 1]
        * [Home Remedy / Care Tip]
        
        ---
        *Disclaimer: This is an AI analysis. Please consult a doctor for a real prescription.*
        """
        
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"

def get_chat_response(user_question, image, disease_context):
    """
    Handles follow-up questions in the chat with smart intent analysis and visual verification.
    """
    try:
        # Use your available Gemini model
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        You are an intelligent Dermatologist Assistant.
        
        --- INPUT DATA ---
        1. **User Image:** (Attached)
        2. **Automated YOLO Detection:** "{disease_context}" (This is a preliminary AI guess. It might be wrong.)
        3. **User Query:** "{user_question}"
        
        --- YOUR INSTRUCTIONS ---
        1. **Analyze the User's Intent First:** - If they ask "Is this correct?" or "Are you sure?", look at the image and agree or disagree based on your own visual analysis (Gemini Vision).
           - If they ask "Tell me the name", give ONLY the name.
           - If they ask about a DIFFERENT disease, answer about that disease. Do not force the conversation back to "{disease_context}" if the user has moved on.
           
        2. **Visual Verification:** - Trust your own eyes more than the YOLO detection. 
           - If the image clearly shows "Hives" but YOLO said "Eczema", politely correct it in your answer.

        3. **Response Style:**
           - Be direct and conversational. 
           - Do NOT repeat the disclaimer in every single message.
           - If the user asks for a short answer, keep it under 2 sentences.
        """
        
        # Send text + image to Gemini
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error connecting to Google Gemini: {e}"

# --- SIDEBAR (Controls) ---
with st.sidebar:
    st.header("🩺 Skin Scanner")
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    
    # === ANALYZE BUTTON LOGIC ===
    if uploaded_file and st.button('🚀 Analyze Condition', type="primary"):
        with st.spinner('1. Scanning pixels (YOLO)...'):
            # Step A: Image Setup
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            
            # Step B: Run YOLO
            model = load_yolo_model()
            results = model.predict(image)
            r = results[0]
            disease_name = r.names[r.probs.top1]
            confidence = r.probs.top1conf.item()
            st.session_state.detected_disease = disease_name
        
        with st.spinner('2. Generating Medical Report (Gemini)...'):
            # Step C: Generate the Detailed Report
            initial_report = generate_initial_report(image, disease_name)
            
            # --- THE FIX: Create the "Analysis Result" Header ---
            analysis_header = f"""
### 🔬 Analysis Result
**YOLO Detection:** {disease_name}
**Confidence:** {confidence*100:.1f}%

---
"""
            # Combine the Header + The Gemini Report
            full_response = analysis_header + initial_report
            
            # Step D: Save this COMBINED text as the first message
            st.session_state.messages = [
                {"role": "assistant", "content": full_response}
            ]
            st.rerun() # Refresh to show the report

    st.divider()
    if st.session_state.detected_disease:
        st.success(f"YOLO Detected: **{st.session_state.detected_disease}**")

# --- MAIN CHAT INTERFACE ---
st.title("SkinCare AI Assistant")

# 1. Display Chat History (This will now include the Initial Report)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input for Follow-up Questions
if prompt := st.chat_input("Ask a follow-up question (e.g., 'Is it contagious?')..."):
    if st.session_state.current_image:
        # User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Dr. AI is typing..."):
                response = get_chat_response(prompt, st.session_state.current_image, st.session_state.detected_disease)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload an image in the sidebar first!")
