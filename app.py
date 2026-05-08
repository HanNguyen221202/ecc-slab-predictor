import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ECC-Slab Predictor", layout="wide")
st.title("Artificial Neural Network (ANN) for Punching Shear Prediction of ECC-Strengthened Flat Slabs")
st.markdown("##### **A Research Product by:** Dr. Cong-Luyen Nguyen | Ngoc Han Nguyen & Duc Nhan Hoang")
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

# 3. MAIN PAGE INPUT PARAMETERS (Thanh trượt mở rộng)
st.markdown("---")

# Tạo thanh trượt bằng st.expander
with st.expander("⚙️ CẤU HÌNH THAM SỐ ĐẦU VÀO (Bấm để Thu gọn / Mở rộng)", expanded=True):
    
    # Tạo 3 cột bên trong thanh trượt
    col1, col2, col3 = st.columns(3)
    
    # ================= CỘT 1 =================
    with col1:
        st.subheader("1. Thickness & Effective Depth")
        tc = st.slider("Concrete thickness - tc (mm)", 100, 160, 100)
        tECC = st.slider("ECC thickness - tECC (mm)", 30, 90, 30)
        
        cover = 15
        d = tc - cover
        st.info(f"💡 Slab's effective height (d) auto-calculated: **{d} mm**")

        st.subheader("4. ECC Layer Properties")
        fc_ECC = st.slider("ECC compressive strength - f'c,ECC (MPa)", 31, 60, 31)
        Ec_ECC = int(np.interp(fc_ECC, [31, 45, 60], [14000, 17000, 20000]))
        st.info(f"💡 Elastic modulus of ECC - Ec,ECC auto-calculated: **{Ec_ECC} MPa**")

    # ================= CỘT 2 =================
    with col2:
        st.subheader("2. Geometry Configuration")
        c1 = st.slider("Column's short side dimension - c1 (mm)", 150, 250, 150)
        c2_c1 = st.slider("Long-to-short side dimension ratio - c2/c1", 1.0, 1.67, 1.0)
        
        L_d = round(950 / d, 3)
        st.info(f"💡 Shear span to effective depth ratio (L/d) auto-calculated: **{L_d}**")
        
        alpha_s = st.selectbox("Loading location - \u03B1s (2: Corner, 3: Edge, 4: Interior)", [2.0, 3.0, 4.0], index=2)

    # ================= CỘT 3 =================
    with col3:
        st.subheader("3. Normal Concrete (NC) Properties")
        fc_c = st.slider("Concrete compressive strength - f'c,c (MPa)", 30, 60, 30)
        Ec_c = int(4700 * np.sqrt(fc_c))
        st.info(f"💡 Elastic modulus of concrete - Ec,c auto-calculated: **{Ec_c} MPa**")

        st.subheader("5. Reinforcement Details")
        fy = st.slider("Rebar yield strength - fy (MPa)", 456, 750, 456)
        mu = st.slider("Reinforcement ratio - \u03BC (%)", 1.227, 2.454, 1.227, step=0.001)

    st.markdown("---")
    
    # Nút bấm ĐƯỢC ĐẶT TRONG st.expander, ngang hàng với dòng col1, col2, col3
    run_button = st.button("🚀 RUN PREDICTION", use_container_width=True)

# 4. DATA PROCESSING & PREDICTION (Nằm ngoài st.expander để kết quả in ra bên dưới thanh trượt)
input_data = np.array([[d, c1, c2_c1, L_d, alpha_s, fc_c, Ec_c, fc_ECC, Ec_ECC, tc, tECC, fy, mu]])

if run_button:
    with st.spinner("Analyzing data..."):
        input_scaled = (input_data - X_min) / (X_max - X_min)
        prediction_norm = model.predict(input_scaled)
        prediction_real = prediction_norm[0][0] * (y_max - y_min) + y_min
        
        st.success("Prediction Completed!")
        
        # Phần in kết quả
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            MAE_error = 4.80
            st.markdown("<p style='font-size: 14px; margin-bottom: 0px; color: #FAFAFA;'>Predicted Punching Shear Capacity (Vp)</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='display: flex; align-items: baseline; gap: 8px; margin-bottom: 0px;'>"
                f"<span style='font-size: 40px; font-weight: bold;'>{prediction_real:.2f} kN</span>"
                f"<span style='font-size: 16px; color: #A5A5A5;'>± {MAE_error} kN Expected Error (MAE)</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("<p style='font-size: 14px; color: #09AB3B; margin-top: 5px;'>↑ ANN Model (R² = 0.99)</p>", unsafe_allow_html=True)
        with col_res2:
            st.info(f"Total Slab Thickness: {tc + tECC:.1f} mm")
            
        st.markdown("---")
        st.markdown("### Structural Configuration & Explainable AI (SHAP)")
        
        image_path = os.path.join(BASE_DIR, 'slab_shap_info.png')
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True, caption="Fig 1. Geometric parameters and SHAP-based feature importance analysis")
        else:
            st.warning("⚠️ Không tìm thấy file ảnh 'slab_shap_info.png' trong thư mục.")
