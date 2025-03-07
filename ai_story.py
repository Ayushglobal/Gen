import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64
import tempfile

# Custom styling
st.markdown("""
<style>
    .story-container {
        border: 2px solid #4a4a4a;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #1a1a1a;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@st.cache_resource
def load_model():
    return ChatOllama(
        model="llava",  # Vision-Language model
        base_url="http://localhost:11434",
        temperature=0.7,
        request_timeout=300
    )

llm = load_model()

st.title("📸 Visual Story Creator")
st.markdown("### Transform Memories into Illustrated Stories")

with st.form("story_form"):
    # Image upload section
    uploaded_images = st.file_uploader(
        "Upload your memory photos:",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    
    # Story context
    story_context = st.text_area(
        "Add context about these photos:",
        placeholder="Describe people, events, or special moments in these photos...",
        height=150
    )
    
    # Style selection
    story_style = st.selectbox(
        "Story Style:",
        ["Nostalgic", "Humorous", "Dramatic", "Poetic"]
    )
    
    generate_btn = st.form_submit_button("Create Story")

if generate_btn:
    if not uploaded_images:
        st.warning("Please upload at least one photo!")
        st.stop()
        
    with st.spinner("🔍 Analyzing photos and crafting story..."):
        try:
            # Process images
            image_descriptions = []
            temp_files = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded images to temp files
                for idx, img_file in enumerate(uploaded_images):
                    temp_path = f"{temp_dir}/img_{idx}.{img_file.type.split('/')[-1]}"
                    with open(temp_path, "wb") as f:
                        f.write(img_file.getbuffer())
                    temp_files.append(temp_path)
                
                # Create image grid
                st.markdown("<div class='image-grid'>", unsafe_allow_html=True)
                for path in temp_files:
                    st.image(path, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Prepare vision prompt
                vision_prompt = [
                    {"type": "text", "text": f"Create a {story_style.lower()} story based on these images and context: {story_context}"}
                ]
                
                # Add images to prompt
                for path in temp_files:
                    vision_prompt.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encode_image(path)}"
                    })
                
                # Generate story
                response = llm.invoke(vision_prompt)
                
                # Display story
                st.markdown("<div class='story-container'>", unsafe_allow_html=True)
                st.markdown("### Your Visual Story")
                st.markdown(response.content)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating story: {str(e)}")

# Sidebar instructions
with st.sidebar:
    st.markdown("## 📌 How to Use:")
    st.markdown("""
    1. Upload 1-5 memory photos
    2. Add context about the photos
    3. Choose story style
    4. Click 'Create Story'
    """)
    st.markdown("## 🖼️ Supported Formats:")
    st.markdown("- JPEG, PNG images\n- Max 5 photos\n- 2MB per photo limit")