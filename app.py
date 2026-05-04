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
# Đưa tc và tECC lên đầu tiên để làm cơ sở tính toán
tc = st.sidebar.slider("Concrete thickness - tc (mm)", 100.0, 160.0, 100.0)
tECC = st.sidebar.slider("ECC thickness - tECC (mm)", 30.0, 90.0, 30.0)

# Khóa cứng lớp bảo vệ và tự động tính d
cover = 15.0 
d = tc - cover # Tổng chiều dày trừ đi lớp bảo vệ (có thể sửa nếu công thức của bạn là d = tc - cover)
st.sidebar.info(f"💡 Slab's effective height (d) auto-calculated: **{d:.1f} mm**")

st.sidebar.subheader("2. Geometry Configuration")
c1 = st.sidebar.slider("Column's short side dimension - c1 (mm)", 150.0, 250.0, 150.0)
c2_c1 = st.sidebar.slider("Long-to-short side dimension ratio - c2/c1", 1.0, 1.67, 1.0)
L_d = st.sidebar.slider("Shear span to effective depth ratio - L/d", 6.5, 11.2, 8.9)
alpha_s = st.sidebar.selectbox("Loading location - αs (2: Corner, 3: Edge, 4: Interior)", [2.0, 3.0, 4.0], index=2)

st.sidebar.subheader("3. Normal Concrete (NC) Properties")
fc_c = st.sidebar.slider("Concrete compressive strength - f'c,c (MPa)", 30.0, 60.0, 30.0)
Ec_c = st.sidebar.slider("Elastic modulus of concrete - Ec,c (MPa)", 25000.0, 37000.0, 28946.0)

st.sidebar.subheader("4. ECC Layer Properties")
fc_ECC = st.sidebar.slider("ECC compressive strength - f'c,ECC (MPa)", 31.0, 60.0, 45.0)
Ec_ECC = st.sidebar.slider("Elastic modulus of ECC - Ec,ECC (MPa)", 14000.0, 20000.0, 17000.0)

st.sidebar.subheader("5. Reinforcement Details")
fy = st.sidebar.slider("Rebar yield strength - fy (MPa)", 456.0, 750.0, 494.0)
mu = st.sidebar.slider("Reinforcement ratio - μ (%)", 1.2, 2.5, 1.49)

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
