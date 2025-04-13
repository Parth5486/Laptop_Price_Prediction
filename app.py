import streamlit as st
import pickle
import numpy as np

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Title and Subheading
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💻 Laptop Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("Welcome! Fill in the specs of your laptop below, and I'll predict the estimated price for you. 🎯")

st.markdown("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
    laptop_type = st.selectbox('💼 Type', df['TypeName'].unique())
    ram = st.selectbox('🧠 RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('⚖️ Weight of the Laptop (kg)')
    touchscreen = st.selectbox('🖱️ Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('📺 IPS Display', ['No', 'Yes'])

with col2:
    screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('🖥️ Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    cpu = st.selectbox('🧩 CPU Brand', df['Cpu brand'].unique())
    hdd = st.selectbox('💽 HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('⚡ SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('🎮 GPU Brand', df['Gpu brand'].unique())
    os = st.selectbox('🖥️ Operating System', df['os'].unique())

# Prediction
st.markdown("---")
if st.button('🔍 Predict Price'):
    # Logic untouched
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    query = query.reshape(1, -1)

    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"💰 The predicted price of this configuration is **₹{predicted_price}**")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")

