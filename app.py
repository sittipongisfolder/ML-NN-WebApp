import streamlit as st
import os
import numpy as np
import joblib


# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML & Neural Network Web App", layout="wide")


# Sidebar สำหรับเลือกหน้า
st.sidebar.title("Intelligent System Project")
st.sidebar.caption("Sittipong Wongsuwan 6404062636544")
st.sidebar.markdown('<p style="font-size:20px; font-weight:bold;">MENU</p>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "Data Preprocessing and Algorithms", "Model  & Features", "ML Demo", "NN Demo"])

# แสดงเนื้อหาตามหน้าที่เลือก
if page == "Home":
    exec(open("page/Home.py", encoding="utf-8").read())

elif page == "Data Preprocessing and Algorithms":
    exec(open("page/data_preprocessing.py", encoding="utf-8").read())

elif page == "Model  & Features":
    exec(open("page/model_ex.py", encoding="utf-8").read())

elif page == "ML Demo":
    exec(open("page/demo_ml.py", encoding="utf-8").read())

elif page == "NN Demo":
    exec(open("page/demo_nn.py", encoding="utf-8").read())