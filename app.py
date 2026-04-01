import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
from PIL import Image, ImageFilter, ImageStat
import io
import base64
from fpdf import FPDF
import tempfile
import re
import os
from dotenv import load_dotenv
import httpx
from skimage import color, filters, measure, morphology
from skimage.transform import resize
import colorsys

# Load environment variables
load_dotenv()

# Page config must be the first Streamlit command
st.set_page_config(page_title="Smart Crop Advisor", layout="wide", page_icon="🌾")

# Initialize Groq client with API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # For Streamlit Cloud, also check secrets
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY not found. Please add it to .env file or Streamlit secrets.")
    st.stop()

# Initialize Groq client with custom http client to avoid proxy issues
try:
    http_client = httpx.Client(timeout=60.0)
    client = Groq(api_key=api_key, http_client=http_client)
except Exception as e:
    st.error(f"Error initializing Groq client: {str(e)}")
    client = None

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
        'area': r'area:?\s*([\d.]+)', 
        'soil_type': r'soil type:?\s*(\w+)', 
        'ph': r'ph:?\s*([\d.]+)',
        'nitrogen': r'nitrogen:?\s*([\d.]+)', 
        'humidity': r'humidity:?\s*([\d.]+)', 
        'moisture': r'moisture:?\s*([\d.]+)',
        'temperature': r'temp:?\s*([\d.]+)', 
        'past_crop': r'past crop:?\s*(\w+)', 
        'season': r'season:?\s*(\w+)',
        'location': r'location:?\s*([\w\s,]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text_content, re.IGNORECASE)
        if match:
            data[key] = match.group(1)
        else:
            data[key] = None
    return data

# Function to analyze crop image using PIL and scikit-image (no OpenCV)
def analyze_crop_image(image):
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Basic image analysis
        if len(img_array.shape) == 3:
            # Convert RGB to HSV using colorsys for each pixel
            h, w, _ = img_array.shape
            hsv_image = np.zeros((h, w, 3))
            
            for i in range(h):
                for j in range(w):
                    r, g, b = img_array[i, j, 0]/255.0, img_array[i, j, 1]/255.0, img_array[i, j, 2]/255.0
                    hsv_image[i, j, 0], hsv_image[i, j, 1], hsv_image[i, j, 2] = colorsys.rgb_to_hsv(r, g, b)
            
            # Disease detection based on color analysis
            # Yellow/brown colors indicate disease
            yellow_mask = (hsv_image[:, :, 0] > 0.08) & (hsv_image[:, :, 0] < 0.15) & (hsv_image[:, :, 1] > 0.3)
            brown_mask = (hsv_image[:, :, 0] > 0.05) & (hsv_image[:, :, 0] < 0.12) & (hsv_image[:, :, 1] > 0.4) & (hsv_image[:, :, 2] < 0.6)
            
            # Calculate disease percentage
            yellow_percentage = np.sum(yellow_mask) / (h * w) * 100
            brown_percentage = np.sum(brown_mask) / (h * w) * 100
            disease_percentage = yellow_percentage + brown_percentage
            
            # Calculate green intensity (health indicator)
            green_channel = img_array[:, :, 1]
            green_intensity = np.mean(green_channel) / 255.0
            
            # Texture analysis using standard deviation
            texture_score = np.std(green_channel)
            
            if disease_percentage > 20:
                return "⚠️ Severe disease detected - Immediate action required", disease_percentage
            elif disease_percentage > 10:
                return "🟡 Moderate disease symptoms detected - Treatment recommended", disease_percentage
            elif disease_percentage > 3:
                return "🟢 Minor issues detected - Monitor closely", disease_percentage
            else:
                # Additional health check based on green intensity
                if green_intensity < 0.4:
                    return "🟡 Crop appears stressed - Check nutrients", disease_percentage
                return "✅ Crop appears healthy - Good condition", disease_percentage
                
        return "Unable to analyze image format", 0
    except Exception as e:
        return f"Error in analysis: {str(e)}", 0

# Function to calculate health metrics from image
def calculate_health_metrics(image):
    """Calculate various health metrics from the crop image"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) != 3:
            return {}
        
        # Calculate color metrics
        r_channel = img_array[:, :, 0].mean()
        g_channel = img_array[:, :, 1].mean()
        b_channel = img_array[:, :, 2].mean()
        
        # Greenness index (normalized difference vegetation index approximation)
        greenness = (g_channel - r_channel) / (g_channel + r_channel + 1e-6)
        
        # Calculate texture (variation)
        texture = np.std(img_array[:, :, 1])
        
        return {
            'greenness': greenness,
            'texture_variation': texture,
            'red_intensity': r_channel,
            'green_intensity': g_channel,
            'blue_intensity': b_channel
        }
    except:
        return {}

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
    
    Format response with clear sections and recommendations. Keep the response concise and actionable."""
    
    try:
        response = client.chat.completions.create(
            model=model_choice,
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
        try:
            nitrogen_val = float(farm_data.get('nitrogen', 50))
            soil_components = pd.DataFrame({
                'Component': ['Nitrogen', 'Phosphorus', 'Potassium', 'Organic Matter'],
                'Value': [nitrogen_val, 30, 40, 20]
            })
            fig = px.pie(soil_components, values='Value', names='Component', title='Soil Composition')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create soil composition chart: {str(e)}")
    
    with fig_col2:
        try:
            humidity_val = float(farm_data.get('humidity', 60))
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=humidity_val,
                title={'text': "Humidity (%)"},
                delta={'reference': 70},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 30], 'color': "lightcoral"},
                               {'range': [30, 60], 'color': "lightblue"},
                               {'range': [60, 100], 'color': "lightgreen"}]}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create humidity gauge: {str(e)}")

# Main UI with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Farm Input", "📊 Analytics", "🔬 Crop Analysis", "📑 Final Report"])

with tab1:
    st.header("Farm Data Input")
    input_method = st.radio("Choose input method:", ["Upload Text File", "Manual Entry"])
    
    if input_method == "Upload Text File":
        uploaded_file = st.file_uploader("Upload farm data text file", type=['txt'])
        if uploaded_file:
            try:
                content = uploaded_file.read().decode()
                st.session_state.farm_data = parse_farm_data(content)
                st.success("✅ File uploaded successfully!")
                with st.expander("View parsed data"):
                    st.json(st.session_state.farm_data)
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")
    else:
        with st.form("manual_input"):
            col1, col2, col3 = st.columns(3)
            with col1:
                area = st.number_input("Area (acres)", value=5.0, step=0.5)
                soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silt", "Peaty", "Chalky"])
                ph = st.slider("pH Value", 0.0, 14.0, 7.0, 0.1)
            with col2:
                nitrogen = st.number_input("Nitrogen (kg/ha)", value=50, step=10)
                humidity = st.slider("Humidity (%)", 0, 100, 60)
                moisture = st.slider("Moisture (%)", 0, 100, 50)
            with col3:
                temp = st.slider("Temperature (°C)", -10, 50, 25)
                past_crop = st.text_input("Past Crop", "Wheat")
                season = st.selectbox("Season", ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"])
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
        try:
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            weather_data = pd.DataFrame({
                'Date': dates,
                'Temperature': np.random.normal(25, 5, 30),
                'Rainfall': np.random.exponential(5, 30),
                'Humidity': np.random.normal(60, 10, 30)
            })
            
            fig = px.line(weather_data, x='Date', y=['Temperature', 'Humidity'], title='Weather Trends (Simulated)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create weather trend chart: {str(e)}")
    else:
        st.warning("⚠️ Please input farm data first in the Farm Input tab")

with tab3:
    st.header("Crop Disease Analysis")
    st.info("Upload a crop image to detect potential diseases and get treatment recommendations.")
    uploaded_image = st.file_uploader("Upload crop image for disease detection", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Crop", width=300)
            
            # Display image metrics
            if st.button("Analyze Crop"):
                with st.spinner("Analyzing crop image..."):
                    # Get analysis result
                    result, percentage = analyze_crop_image(image)
                    
                    # Calculate additional health metrics
                    health_metrics = calculate_health_metrics(image)
                    
                    st.subheader("Analysis Result")
                    
                    if "healthy" in result.lower() or "good condition" in result.lower():
                        st.success(result)
                    elif "minor" in result.lower() or "stressed" in result.lower():
                        st.warning(result)
                    else:
                        st.error(result)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Disease Probability", f"{percentage:.1f}%")
                    with col2:
                        st.metric("Greenness Index", f"{health_metrics.get('greenness', 0):.2f}")
                    with col3:
                        st.metric("Texture Variation", f"{health_metrics.get('texture_variation', 0):.1f}")
                    
                    st.progress(min(percentage/100, 1.0))
                    
                    # Get LLM advice for disease
                    if st.session_state.farm_data and client:
                        try:
                            prompt = f"""Based on {percentage:.1f}% disease detection in {st.session_state.farm_data.get('past_crop', 'crop')}, 
                            with greenness index {health_metrics.get('greenness', 0):.2f}, 
                            provide:
                            1. Likely disease identification
                            2. Treatment recommendations
                            3. Preventive measures
                            Keep response concise and actionable."""
                            advice = get_recommendations({**st.session_state.farm_data, 'disease_percentage': percentage})
                            st.info(advice)
                        except Exception as e:
                            st.warning(f"Could not get AI advice: {str(e)}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab4:
    st.header("Final Comprehensive Report")
    
    if st.session_state.farm_data:
        if st.button("Generate Complete Report"):
            with st.spinner("Generating AI-powered report..."):
                # Get LLM recommendations
                if not st.session_state.analysis:
                    st.session_state.analysis = get_recommendations(st.session_state.farm_data)
                
                # Display report
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📋 Input Summary")
                    for key, value in st.session_state.farm_data.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    # Download PDF
                    if st.button("📥 Download PDF Report"):
                        try:
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
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                
                with col2:
                    st.subheader("🤖 AI Recommendations & Analysis")
                    st.write(st.session_state.analysis)
                    
                    # Risk assessment gauge
                    try:
                        risk_score = 0
                        ph_val = float(st.session_state.farm_data.get('ph', 7))
                        if ph_val < 5.5 or ph_val > 8.5:
                            risk_score += 30
                        
                        nitrogen_val = float(st.session_state.farm_data.get('nitrogen', 50))
                        if nitrogen_val < 20:
                            risk_score += 20
                        
                        moisture_val = float(st.session_state.farm_data.get('moisture', 50))
                        if moisture_val < 30:
                            risk_score += 20
                        
                        temp_val = float(st.session_state.farm_data.get('temperature', 25))
                        if temp_val > 35:
                            risk_score += 20
                        
                        risk_score = min(risk_score, 100)
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_score,
                            title={'text': "Risk Assessment Score"},
                            gauge={'axis': {'range': [0, 100]},
                                  'bar': {'color': "red" if risk_score > 60 else "orange" if risk_score > 30 else "green"},
                                  'steps': [{'range': [0, 30], 'color': "lightgreen"},
                                           {'range': [30, 60], 'color': "yellow"},
                                           {'range': [60, 100], 'color': "lightcoral"}]}))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create risk assessment: {str(e)}")
    else:
        st.warning("⚠️ Please input farm data first in the Farm Input tab")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Smart Crop Advisor v1.0 - AI Powered Farming Intelligence</p>", unsafe_allow_html=True)
