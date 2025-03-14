import streamlit as st
import pandas as pd
import joblib
import numpy as np
import catboost
import os
from catboost import CatBoostClassifier


@st.cache_resource
def load_model():
    model_path = "../models/catboost_model.cbm"
    
    if not os.path.exists(model_path):
        st.error(f"🚨 ไม่พบไฟล์โมเดล: {model_path}")
        return None
    
    model = CatBoostClassifier()
    model.load_model(model_path, format="cbm")  # โหลดโมเดลโดยตรงจากไฟล์ .cbm
    return model

model = load_model()

# Mapping Class Number -> ชื่อสัตว์
class_mapping = {
    1: "สัตว์เลี้ยงลูกด้วยนม (Mammal)",
    2: "นก (Bird)",
    3: "สัตว์เลื้อยคลาน (Reptile)",
    4: "ปลา (Fish)",
    5: "สัตว์ครึ่งบกครึ่งน้ำ (Amphibian)",
    6: "แมลง (Insect)",
    7: "สัตว์ไม่มีกระดูกสันหลัง (Invertebrate)"
}

# ฟีเจอร์ของ Zoo Dataset พร้อมคำอธิบายภาษาไทย
features = {
    "hair": "มีขนหรือไม่?",
    "feathers": "มีขนแบบนกหรือไม่?",
    "eggs": "ออกไข่หรือไม่?",
    "milk": "สามารถผลิตน้ำนมได้หรือไม่?",
    "airborne": "สามารถบินได้หรือไม่?",
    "aquatic": "เป็นสัตว์น้ำหรือไม่?",
    "predator": "เป็นสัตว์นักล่าหรือไม่?",
    "toothed": "มีฟันหรือไม่?",
    "backbone": "มีแกนกระดูกสันหลังหรือไม่?",
    "breathes": "หายใจด้วยปอดหรือไม่?",
    "venomous": "มีพิษหรือไม่?",
    "fins": "มีครีบหรือไม่?",
    "legs": "จำนวนขา",
    "tail": "มีหางหรือไม่?",
    "domestic": "เป็นสัตว์เลี้ยงหรือไม่?",
    "catsize": "มีขนาดใกล้เคียงกับแมวหรือไม่?",
}

st.title("🔍 ทดสอบระบบจำแนกประเภทสัตว์")
st.write("กรอกข้อมูลเกี่ยวกับสัตว์เพื่อให้โมเดลพยากรณ์ประเภทของสัตว์ 🦁🐦🐠")

# จัด Layout ให้ดูเรียบร้อยขึ้น
col1, col2 = st.columns(2)

user_input = {}

# Loop ผ่าน features และแบ่งเป็น 2 คอลัมน์
for i, (feature, desc) in enumerate(features.items()):
    if feature == "legs":  # จำนวนขา ให้ใช้ selectbox
        user_input[feature] = st.selectbox(desc, [0, 2, 4, 5, 6, 8])
    else:
        col = col1 if i % 2 == 0 else col2
        user_input[feature] = col.radio(desc, ["ไม่ใช่", "ใช่"], index=0)

# แปลงข้อมูลเป็น 0 และ 1
input_data = [1 if user_input[f] == "ใช่" else 0 for f in features if f != "legs"]
input_data.insert(12, user_input["legs"])  # แทรกค่าของ 'legs' ในตำแหน่งที่ถูกต้อง

# แปลงเป็น DataFrame
input_df = pd.DataFrame([input_data], columns=features.keys())

# ปุ่ม Predict ให้ดูชัดเจนขึ้น
st.markdown("<br>", unsafe_allow_html=True)  # เพิ่มระยะห่าง
if model:
    if st.button("🔮 ทำนายประเภทของสัตว์"):
        prediction = model.predict(input_df)

        # ✅ ใช้ item() เพื่อแปลงเป็น int
        if isinstance(prediction, (np.ndarray, list)):
            class_id = int(prediction[0])
        else:
            class_id = int(prediction)

        # ดึงชื่อสัตว์จาก class_mapping
        animal_class = class_mapping.get(class_id, "ประเภทไม่ทราบ")
        
        st.success(f"🎯 สัตว์ที่คาดการณ์ได้: **{animal_class}**")
