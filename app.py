import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ECC-Slab Predictor", layout="wide")
st.title("🏗️ Artificial Neural Network (ANN) for Punching Shear Prediction of ECC-Strengthened Flat Slabs")
st.markdown("---")

# 2. LOAD MODEL AND SCALER PARAMS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'ann_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'scaler_params.npz')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Cannot find: {scaler_path}")
        
    model = load_model(model_path, compile=False) 
    scaler_params = np.load(scaler_path)
    
    X_min = scaler_params['X_min']
    X_max = scaler_params['X_max']
    y_min = scaler_params['y_min']
    y_max = scaler_params['y_max']
    
    return model, X_min, X_max, y_min, y_max

try:
    model, X_min, X_max, y_min, y_max = load_assets()
except Exception as e:
    st.error(f"INITIALIZATION ERROR: {e}")
    st.stop()

# 3. SIDEBAR (INPUT PARAMETERS)
st.sidebar.header("📥 INPUT PARAMETERS")

st.sidebar.subheader("1. Thickness & Effective Depth")
# Đã xóa .0 để lấy số nguyên
tc = st.sidebar.slider("Concrete thickness - tc (mm)", 100, 160, 100)
tECC = st.sidebar.slider("ECC thickness - tECC (mm)", 30, 90, 30)

cover = 15 # Lớp bảo vệ số nguyên
d = tc - cover
st.sidebar.info(f"💡 Slab's effective height (d) auto-calculated: **{d} mm**")

st.sidebar.subheader("2. Geometry Configuration")
c1 = st.sidebar.slider("Column's short side dimension - c1 (mm)", 150, 250, 150)
c2_c1 = st.sidebar.slider("Long-to-short side dimension ratio - c2/c1", 1.0, 1.67, 1.0)

# XÓA SLIDER L/d. Tính tự động vì L = 950 mm (cố định theo bài báo)
L_d = round(950 / d, 3) 
st.sidebar.info(f"💡 Shear span to effective depth ratio (L/d) auto-calculated: **{L_d}**")

alpha_s = st.sidebar.selectbox("Loading location - αs (2: Corner, 3: Edge, 4: Interior)", [2.0, 3.0, 4.0], index=2)

st.sidebar.subheader("3. Normal Concrete (NC) Properties")
fc_c = st.sidebar.slider("Concrete compressive strength - f'c,c (MPa)", 30, 60, 30)
# Tự động tính Ec_c
Ec_c = int(4700 * np.sqrt(fc_c))
st.sidebar.info(f"💡 Elastic modulus of concrete - Ec,c auto-calculated: **{Ec_c} MPa**")

st.sidebar.subheader("4. ECC Layer Properties")
# Đưa default về 31 để khớp với sàn U3 gốc
fc_ECC = st.sidebar.slider("ECC compressive strength - f'c,ECC (MPa)", 31, 60, 31) 
# Tự động tính Ec_ECC
Ec_ECC = int(np.interp(fc_ECC, [31, 45, 60], [14000, 17000, 20000]))
st.sidebar.info(f"💡 Elastic modulus of ECC - Ec,ECC auto-calculated: **{Ec_ECC} MPa**")

st.sidebar.subheader("5. Reinforcement Details")
# Đưa default về 456 để khớp với sàn U3 gốc
fy = st.sidebar.slider("Rebar yield strength - fy (MPa)", 456, 750, 456) 

# SỬA LỖI NGHIÊM TRỌNG: Thiết lập lại min/max và step cho mu để khớp tuyệt đối với Dataset!
mu = st.sidebar.slider("Reinforcement ratio - μ (%)", 1.227, 2.454, 1.227, step=0.001)

# 4. DATA PROCESSING & PREDICTION
input_data = np.array([[d, c1, c2_c1, L_d, alpha_s, fc_c, Ec_c, fc_ECC, Ec_ECC, tc, tECC, fy, mu]])

if st.sidebar.button("🚀 RUN PREDICTION", use_container_width=True):
    with st.spinner("Analyzing data..."):
        input_scaled = (input_data - X_min) / (X_max - X_min)
        prediction_norm = model.predict(input_scaled)
        prediction_real = prediction_norm[0][0] * (y_max - y_min) + y_min
        
        st.success("Prediction Completed!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Punching Shear Capacity (Vp)", value=f"{prediction_real:.2f} kN", delta="ANN Model (R²=0.99)")
        with col2:
            st.info(f"Total Slab Thickness: {tc + tECC:.1f} mm")
            
        st.markdown("---")
        st.markdown("### 📊 Structural Configuration & Explainable AI (SHAP)")
        
        image_path = os.path.join(BASE_DIR, 'slab_shap_info.png')
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True, caption="Fig 1. Geometric parameters and SHAP-based feature importance analysis")
        else:
            st.warning("⚠️ Không tìm thấy file ảnh 'slab_shap_info.png' trong thư mục.")
