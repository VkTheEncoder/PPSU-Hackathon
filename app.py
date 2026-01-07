import streamlit as st
from ultralytics import YOLO
from PIL import Image
from groq import Groq

# --- CONFIGURATION ---
# Replace with your actual key
GROQ_API_KEY = "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"

# --- Page Configuration (Gemini Style) ---
st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🩺",
    layout="centered" # "Centered" looks more like a chat app than "Wide"
)

# --- CSS Hack for "Clean Look" ---
# This removes the massive white space at the top of the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "detected_disease" not in st.session_state:
    st.session_state.detected_disease = None

# --- Load Models ---
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

def get_ai_response(user_question, disease_context):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = f"""
        You are an expert Dermatologist AI Assistant.
        The user has uploaded a skin image which was detected as: '{disease_context}'.
        Your Goal: Answer the user's questions specifically about '{disease_context}'.
        - Keep answers concise, professional, and helpful.
        - Use simple medical terms where possible.
        - Always include a disclaimer if discussing treatments.
        """

        messages_for_api = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.messages:
            if msg["role"] != "system": # Skip system messages in history
                messages_for_api.append({"role": msg["role"], "content": msg["content"]})
        
        messages_for_api.append({"role": "user", "content": user_question})

        chat_completion = client.chat.completions.create(
            messages=messages_for_api,
            model="llama-3.3-70b-versatile", 
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- SIDEBAR (The "Controls") ---
with st.sidebar:
    st.header("🩺 Skin Scanner")
    st.write("Upload an image to start a new consultation.")
    
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    
    # Logic: When user clicks "Analyze", we inject the result into the chat
    if uploaded_file and st.button('🚀 Analyze Condition', type="primary"):
        with st.spinner('Scanning image...'):
            # 1. Run YOLO
            image = Image.open(uploaded_file)
            model = load_yolo_model()
            results = model.predict(image)
            r = results[0]
            disease_name = r.names[r.probs.top1]
            confidence = r.probs.top1conf.item()
            
            # 2. Save Context
            st.session_state.detected_disease = disease_name
            
            # 3. Create the "Analysis" Message
            # We save the image to a buffer to display it in chat history if needed, 
            # but for now, we just tell the user what we found.
            
            analysis_msg = (
                f"### 🔬 Analysis Result\n"
                f"**Detected:** {disease_name}\n"
                f"**Confidence:** {confidence*100:.1f}%\n\n"
                f"I have loaded this diagnosis into my context. You can now ask me questions like:"
                f"\n- *What is the treatment?*"
                f"\n- *Is this dangerous?*"
            )
            
            # 4. Clear old chat and start fresh
            st.session_state.messages = [
                {"role": "assistant", "content": analysis_msg}
            ]
            st.rerun() # Force a refresh to show the new chat

    st.divider()
    if st.session_state.detected_disease:
        st.success(f"Context: **{st.session_state.detected_disease}**")
    else:
        st.info("No active diagnosis.")

# --- MAIN PAGE (The "Chat") ---

st.title("SkinCare AI Assistant")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input (Fixed at bottom like Gemini)
if prompt := st.chat_input("Ask about the condition..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI Response
    if st.session_state.detected_disease:
        with st.chat_message("assistant"):
            with st.spinner("Consulting medical database..."):
                response = get_ai_response(prompt, st.session_state.detected_disease)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Fallback if they chat without uploading
        error_msg = "Please upload and analyze an image in the sidebar first! 👈"
        with st.chat_message("assistant"):
            st.error(error_msg)
