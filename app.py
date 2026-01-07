import streamlit as st
from ultralytics import YOLO
from PIL import Image
from groq import Groq

# --- CONFIGURATION ---
# Replace with your actual key
GROQ_API_KEY = "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"

# --- Page Configuration ---
st.set_page_config(
    page_title="SkinCare AI Chat",
    page_icon="🩺",
    layout="wide"
)

# --- Initialize Session State (Memory) ---
# This keeps the chat history alive even when you click buttons
if "messages" not in st.session_state:
    st.session_state.messages = []

if "detected_disease" not in st.session_state:
    st.session_state.detected_disease = None

# --- Load Models ---
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

def get_ai_response(user_question, disease_context):
    """
    Sends the chat history and context to Groq (Llama 3)
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # System prompt: We tell the AI its role and the context (the disease)
        system_prompt = f"""
        You are an expert Dermatologist AI Assistant.
        The user has uploaded a skin image which was detected as: '{disease_context}'.
        
        Your Goal: Answer the user's questions specifically about '{disease_context}'.
        - If they ask for treatments, give advice relevant to {disease_context}.
        - If they ask if it is contagious, answer based on {disease_context}.
        - Keep answers concise, professional, and helpful.
        - Always include a medical disclaimer if giving specific treatment advice.
        """

        # Prepare the messages list for the API
        messages_for_api = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add the conversation history so the AI remembers previous questions
        for msg in st.session_state.messages:
            messages_for_api.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the user's new current question
        messages_for_api.append({"role": "user", "content": user_question})

        # Get response from Llama 3
        chat_completion = client.chat.completions.create(
            messages=messages_for_api,
            model="llama-3.3-70b-versatile", # Updated to the latest working model
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- Main Layout ---
st.title("🩺 AI Skin Doctor Chat")

col1, col2 = st.columns([1, 1.5], gap="medium")

# === COLUMN 1: IMAGE & DETECTION ===
with col1:
    st.subheader("1. Upload & Scan")
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('🔍 Analyze Image', use_container_width=True):
            with st.spinner('Scanning image...'):
                # 1. Run YOLO
                model = load_yolo_model()
                results = model.predict(image)
                r = results[0]
                disease_name = r.names[r.probs.top1]
                confidence = r.probs.top1conf.item()
                
                # 2. Save to Memory
                st.session_state.detected_disease = disease_name
                
                # 3. Start the Chat with a greeting
                initial_msg = f"I have analyzed the image. \n\n**Detected Condition:** {disease_name} ({confidence*100:.1f}% confidence).\n\nYou can now ask me any questions about this condition (e.g., 'Is it contagious?', 'Home remedies?', 'What causes this?')."
                
                # Clear old chat and add new greeting
                st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                st.rerun() # Refresh to show the chat

# === COLUMN 2: CHAT INTERFACE ===
with col2:
    st.subheader("2. Chat with the AI Doctor")
    
    # Logic: Only show chat if a disease has been detected OR if we just want a general chat
    if st.session_state.detected_disease:
        st.info(f"Context: Discussing **{st.session_state.detected_disease}**")
    else:
        st.info("Please upload and analyze an image to start a specific consultation.")

    # 1. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Chat Input Box
    if prompt := st.chat_input("Ask a follow-up question..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Get AI Response
        if st.session_state.detected_disease:
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt, st.session_state.detected_disease)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Add AI response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Please analyze an image first so I know what we are talking about!")
