import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Minimalist black theme
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #1a1a1a !important;
        color: white !important;
        border: 1px solid #4a4a4a !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    .response-box {
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize LLM with caching
@st.cache_resource
def load_llm():
    return ChatOllama(
        model="mistral",
        base_url="http://localhost:11434",
        temperature=0.3,
        system="You are an expert travel planner specializing in Indian tourism."
    )

llm = load_llm()

st.title("🇮🇳 India Travel Planner")
st.markdown("### AI-Powered Itinerary Generator for Indian Destinations")

# Simplified input form
with st.form("travel_form"):
    destination = st.text_input("Destination (e.g., Jaipur, Goa, Kerala)", 
                              placeholder="Enter Indian destination")
    budget = st.number_input("Total Budget (₹)", min_value=5000, value=25000, step=1000)
    days = st.number_input("Trip Duration (Days)", 3, 14, 5)
    travel_style = st.selectbox("Travel Style", ["Budget", "Mid-range", "Luxury"])
    submitted = st.form_submit_button("Generate Plan")

# Optimized prompt template
PROMPT_TEMPLATE = """Create a {days}-day {style} itinerary for {dest} with ₹{budget} budget focusing on:
- Must-see cultural/historical sites
- Local transportation options (auto, taxi, buses)
- Authentic food experiences
- Budget-friendly hotels (₹ per night)
- Seasonal considerations (current month: June)

Format clearly with daily sections and realistic pricing in INR."""

if submitted:
    with st.spinner("🚩 Creating your Indian travel plan..."):
        try:
            # Direct text generation without external APIs
            chain = (
                ChatPromptTemplate.from_template(PROMPT_TEMPLATE) 
                | llm 
                | StrOutputParser()
            )
            
            response = chain.invoke({
                "dest": destination,
                "budget": budget,
                "days": days,
                "style": travel_style
            })
            
            st.markdown(f'<div class="response-box">{response}</div>', 
                      unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Performance optimization sidebar
with st.sidebar:
    st.markdown("### ⚡ Performance Tips")
    st.markdown("""
    1. Keep Ollama server running
    2. Use shorter destination names
    3. Limit to 7-day itineraries
    4. Close other heavy apps
    """)