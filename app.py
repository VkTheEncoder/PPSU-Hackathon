import streamlit as st
from ultralytics import YOLO
from PIL import Image
from groq import Groq  # Import the brain

# --- CONFIGURATION ---
# PASTE YOUR GROQ API KEY HERE inside the quotes
GROQ_API_KEY = "PASTE_YOUR_GSK_KEY_HERE"

# --- Page Configuration ---
st.set_page_config(
    page_title="SkinCare AI Assistant",
    page_icon="🩺",
    layout="wide"  # Changed to wide for better reading
)

# --- Initialize AI Models ---
@st.cache_resource
def load_yolo_model():
    # Load your BEST trained model
    return YOLO("best.pt")

def get_doctor_advice(disease_name):
    """
    This function sends the disease name to Llama 3 and gets a detailed response.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # This is the prompt we send to the AI
        prompt = f"""
        You are an expert Dermatologist. A patient has been diagnosed with '{disease_name}' by an AI screening tool.
        Please provide a detailed response in the following format:
        
        1. **What is it?** (A simple explanation of the condition)
        2. **Symptoms:** (Common symptoms)
        3. **Potential Treatments:** (Medical and home remedies)
        4. **When to see a doctor:** (Urgent warning signs)
        
        Keep the tone professional, empathetic, and informative. Use bullet points.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192", # We use Llama 3 here
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Llama 3: {e}. Check your API Key."

# --- Main App Interface ---
st.title("🩺 AI Skin Disease Consultant")
st.markdown("### Detect. Understand. Treat.")
st.write("Upload a photo of the skin condition. Our AI will analyze it and provide medical context.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        analyze_button = st.button('🔍 Analyze Condition', use_container_width=True)

# --- Logic: If Button Clicked ---
if uploaded_file and analyze_button:
    with col1:
        with st.spinner('👀 The AI "Eyes" (YOLO) are looking...'):
            model = load_yolo_model()
            results = model.predict(image)
            r = results[0]
            pred_class_index = r.probs.top1
            confidence = r.probs.top1conf.item()
            disease_name = r.names[pred_class_index]

        # Display Detection Result
        if confidence > 0.4: # Only show if confidence is decent
            st.success(f"**Detected:** {disease_name}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
        else:
            st.warning(f"**Potential Match:** {disease_name}")
            st.write(f"Confidence: {confidence * 100:.2f}% (Uncertain)")

    # --- THE NEW PART: LLAMA 3 ---
    with col2:
        st.header("📝 Doctor's Analysis")
        st.write("Generating report using **Llama 3 AI**...")
        
        with st.spinner('🧠 The AI "Brain" (Llama 3) is thinking...'):
            # Call our function to get text
            doctor_response = get_doctor_advice(disease_name)
            
            # Display the text nicely
            st.markdown("---")
            st.markdown(doctor_response)
            st.markdown("---")
            st.warning("⚠️ **Disclaimer:** This is an AI-generated report. It is not a substitute for professional medical advice. Please visit a real doctor.")
