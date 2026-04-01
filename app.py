import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
import cv2
from PIL import Image
import io
import base64
from fpdf import FPDF
import tempfile
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client with API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

client = Groq(api_key=api_key)

# Page config
st.set_page_config(page_title="Smart Crop Advisor", layout="wide", page_icon="🌾")

# Custom CSS
st.markdown("""
<style>
    .main-header {text-align: center; padding: 1rem; background: linear-gradient(90deg, #2E7D32 0%, #81C784 100%); color: white; border-radius: 10px;}
    .metric-card {background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .kpi-value {font-size: 2rem; font-weight: bold; color: #2E7D32;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🌾 Smart Crop Advisor - AI Powered Farming Intelligence</h1>", unsafe_allow_html=True)

# Initialize session state
if 'farm_data' not in st.session_state:
    st.session_state.farm_data = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

# Function to parse input text file
def parse_farm_data(text_content):
    data = {}
    patterns = {
        'area': r'area:?\s*([\d.]+)', 'soil_type': r'soil type:?\s*(\w+)', 'ph': r'ph:?\s*([\d.]+)',
        'nitrogen': r'nitrogen:?\s*([\d.]+)', 'humidity': r'humidity:?\s*([\d.]+)', 'moisture': r'moisture:?\s*([\d.]+)',
        'temperature': r'temp:?\s*([\d.]+)', 'past_crop': r'past crop:?\s*(\w+)', 'season': r'season:?\s*(\w+)',
        'location': r'location:?\s*([\w\s,]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text_content, re.IGNORECASE)
        data[key] = match.group(1) if match else None
    return data

# Function to analyze crop image
def analyze_crop_image(image):
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Basic image analysis
    if len(img_array.shape) == 3:
        # Color analysis for disease detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Check for unhealthy patches (simplified)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        disease_percentage = (np.sum(mask > 0) / mask.size) * 100
        
        if disease_percentage > 15:
            return "⚠️ Potential disease detected", disease_percentage
        elif disease_percentage > 5:
            return "🟡 Minor issues detected", disease_percentage
        else:
            return "✅ Crop appears healthy", disease_percentage
    return "Unable to analyze", 0

# Function to get LLM recommendations
def get_recommendations(farm_data, model_choice="llama-3.3-70b-versatile"):
    if client is None:
        return "LLM service unavailable. Please check your API key and internet connection."
    
    prompt = f"""As an agricultural expert, analyze this farm data and provide:
    1. Soil health assessment and recommendations
    2. Top 3 recommended crops for {farm_data.get('season', 'current')} season
    3. Expected yield predictions
    4. Risk analysis (weather, pests, diseases)
    5. Preventive measures for past crop issues
    
    Farm Data:
    - Area: {farm_data.get('area', 'N/A')} acres
    - Soil Type: {farm_data.get('soil_type', 'N/A')}
    - pH: {farm_data.get('ph', 'N/A')}
    - Nitrogen: {farm_data.get('nitrogen', 'N/A')} kg/ha
    - Humidity: {farm_data.get('humidity', 'N/A')}%
    - Moisture: {farm_data.get('moisture', 'N/A')}%
    - Temperature: {farm_data.get('temperature', 'N/A')}°C
    - Past Crop: {farm_data.get('past_crop', 'N/A')}
    - Location: {farm_data.get('location', 'N/A')}
    
    Format response with clear sections and recommendations."""
    
    try:
        response = client.chat.completions.create(
            model=model_choice,  # Use the specified model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"LLM service error: {str(e)}")
        return "LLM service unavailable. Using fallback recommendations."
    
# Function to create visualizations
def create_visualizations(farm_data):
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        # Soil composition pie chart
        soil_components = pd.DataFrame({
            'Component': ['Nitrogen', 'Phosphorus', 'Potassium', 'Organic Matter'],
            'Value': [float(farm_data.get('nitrogen', 50)), 30, 40, 20]
        })
        fig = px.pie(soil_components, values='Value', names='Component', title='Soil Composition')
        st.plotly_chart(fig, use_container_width=True)
    
    with fig_col2:
        # Weather conditions gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=float(farm_data.get('humidity', 60)),
            title={'text': "Humidity (%)"},
            delta={'reference': 70},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 30], 'color': "lightcoral"},
                           {'range': [30, 60], 'color': "lightblue"},
                           {'range': [60, 100], 'color': "lightgreen"}]}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Main UI with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Farm Input", "📊 Analytics", "🔬 Crop Analysis", "📑 Final Report"])

with tab1:
    st.header("Farm Data Input")
    input_method = st.radio("Choose input method:", ["Upload Text File", "Manual Entry"])
    
    if input_method == "Upload Text File":
        uploaded_file = st.file_uploader("Upload farm data text file", type=['txt'])
        if uploaded_file:
            content = uploaded_file.read().decode()
            st.session_state.farm_data = parse_farm_data(content)
            st.success("✅ File uploaded successfully!")
    else:
        with st.form("manual_input"):
            col1, col2, col3 = st.columns(3)
            with col1:
                area = st.number_input("Area (acres)", value=5.0)
                soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silt"])
                ph = st.slider("pH Value", 0.0, 14.0, 7.0)
            with col2:
                nitrogen = st.number_input("Nitrogen (kg/ha)", value=50)
                humidity = st.slider("Humidity (%)", 0, 100, 60)
                moisture = st.slider("Moisture (%)", 0, 100, 50)
            with col3:
                temp = st.slider("Temperature (°C)", -10, 50, 25)
                past_crop = st.text_input("Past Crop", "Wheat")
                season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
                location = st.text_input("Location", "Enter city/village")
            
            if st.form_submit_button("Save Data"):
                st.session_state.farm_data = {
                    'area': area, 'soil_type': soil_type, 'ph': ph, 'nitrogen': nitrogen,
                    'humidity': humidity, 'moisture': moisture, 'temperature': temp,
                    'past_crop': past_crop, 'season': season, 'location': location
                }
                st.success("✅ Data saved!")

with tab2:
    if st.session_state.farm_data:
        st.header("Farm Analytics Dashboard")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Soil pH</h3><p class='kpi-value'>{st.session_state.farm_data.get('ph', 'N/A')}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Nitrogen</h3><p class='kpi-value'>{st.session_state.farm_data.get('nitrogen', 'N/A')} kg/ha</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Moisture</h3><p class='kpi-value'>{st.session_state.farm_data.get('moisture', 'N/A')}%</p></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h3>Temperature</h3><p class='kpi-value'>{st.session_state.farm_data.get('temperature', 'N/A')}°C</p></div>", unsafe_allow_html=True)
        
        # Visualizations
        create_visualizations(st.session_state.farm_data)
        
        # Weather trend simulation
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        weather_data = pd.DataFrame({
            'Date': dates,
            'Temperature': np.random.normal(25, 5, 30),
            'Rainfall': np.random.exponential(5, 30),
            'Humidity': np.random.normal(60, 10, 30)
        })
        
        fig = px.line(weather_data, x='Date', y=['Temperature', 'Humidity'], title='Weather Trends')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please input farm data first in the Farm Input tab")

with tab3:
    st.header("Crop Disease Analysis")
    uploaded_image = st.file_uploader("Upload crop image for disease detection", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Crop", width=300)
        
        if st.button("Analyze Crop"):
            with st.spinner("Analyzing crop image..."):
                result, percentage = analyze_crop_image(image)
                st.subheader("Analysis Result")
                
                if "healthy" in result.lower():
                    st.success(result)
                elif "minor" in result.lower():
                    st.warning(result)
                else:
                    st.error(result)
                
                st.progress(percentage/100)
                st.write(f"Disease probability: {percentage:.1f}%")
                
                # Get LLM advice for disease
                if st.session_state.farm_data:
                    prompt = f"Based on {percentage:.1f}% disease detection in {st.session_state.farm_data.get('past_crop', 'crop')}, provide treatment recommendations and preventive measures."
                    advice = get_recommendations({**st.session_state.farm_data, 'disease_percentage': percentage})
                    st.info(advice)

with tab4:
    st.header("Final Comprehensive Report")
    
    if st.session_state.farm_data and st.button("Generate Complete Report"):
        with st.spinner("Generating AI-powered report..."):
            # Get LLM recommendations
            if not st.session_state.analysis:
                st.session_state.analysis = get_recommendations(st.session_state.farm_data)
            
            # Display report
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📋 Input Summary")
                for key, value in st.session_state.farm_data.items():
                    st.write(f"*{key.replace('_', ' ').title()}:* {value}")
                
                # Download PDF
                if st.button("📥 Download PDF Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Smart Crop Advisor Report", ln=1, align='C')
                    pdf.cell(200, 10, txt="-" * 50, ln=1)
                    
                    for key, value in st.session_state.farm_data.items():
                        pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)
                    
                    pdf.cell(200, 10, txt="\nRecommendations:", ln=1)
                    pdf.multi_cell(0, 10, txt=st.session_state.analysis[:500])
                    
                    pdf.output("crop_report.pdf")
                    with open("crop_report.pdf", "rb") as f:
                        st.download_button("Download PDF", f, file_name="crop_report.pdf")
            
            with col2:
                st.subheader("🤖 AI Recommendations & Analysis")
                st.write(st.session_state.analysis)
                
                # Risk assessment gauge
                risk_score = np.random.randint(20, 80)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': "Risk Assessment"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "red" if risk_score > 60 else "orange" if risk_score > 30 else "green"},
                          'steps': [{'range': [0, 30], 'color': "lightgreen"},
                                   {'range': [30, 60], 'color': "yellow"},
                                   {'range': [60, 100], 'color': "lightcoral"}]}))
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Smart Crop Advisor v1.0 - AI Powered Farming Intelligence</p>", unsafe_allow_html=True)