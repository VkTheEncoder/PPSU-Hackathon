import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# --- CONFIGURATION ---
# PASTE YOUR GOOGLE API KEY HERE
GOOGLE_API_KEY = "AIzaSyC1nkc5o4h1oNmyagubXUXWtdHfWrRg2nk"

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

# --- AI FUNCTIONS ---

def generate_initial_report(disease_context):
    """
    Generates the report based ONLY on the YOLO diagnosis (Text-only).
    Gemini does NOT look at the image.
    """
    try:
        # We can use a text-only model or the same flash model (it handles text too)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        You are an expert Dermatologist and you have to use easy english language so every user can understand properly.
        
        Task: Create a structured initial medical report for the patient.
        
        Input Context:
        The patient has been diagnosed with: "{disease_context}"
        (This diagnosis comes from a specialized AI tool. Accept it as 100% accurate.)
        
        Output Format (Use Markdown):
        ## 🩺 Diagnosis Report: {disease_context}
        
        ### 📖 What is it?
        [Brief explanation of {disease_context}]
        
        ### ⚠️ Common Symptoms
        * [Symptom 1 of this disease]
        * [Symptom 2 of this disease]
        
        ### 💊 Suggested Treatments & Care
        * [Treatment 1]
        * [Home Remedy / Care Tip]
        
        ---
        *Disclaimer: This is an AI analysis based on the detected label. Please consult a doctor.*
        """
        
        # CHANGED: We send ONLY the prompt (text), no image.
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"

def get_chat_response(user_question, disease_context):
    """
    Handles chat questions based ONLY on the disease name.
    """
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        You are an intelligent Dermatologist Assistant and you have to use easy english language so every user can understand properly.
        
        Context: The patient has a skin condition diagnosed as: "{disease_context}".
        User Query: "{user_question}"
        
        Instructions:
        1. Answer the user's question specifically regarding "{disease_context}".
        2. Do NOT ask to see the image or try to verify the diagnosis. Assume "{disease_context}" is correct.
	    3. If they ask about a DIFFERENT disease, answer about that disease. Do not force the conversation back to "{disease_context}" if the user has moved on.
        4. Keep answers direct, conversational, and professional.
        5. If the user asks for a short answer, keep it under 2 sentences.
	    6. Do NOT repeat the disclaimer in every single message.
        """
        
        # CHANGED: We send ONLY the prompt (text), no image.
        response = model.generate_content(prompt)
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
            initial_report = generate_initial_report(disease_name)
            
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
            with st.spinner("Thinking..."):
                response = get_chat_response(prompt, st.session_state.detected_disease)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload an image in the sidebar first!")
