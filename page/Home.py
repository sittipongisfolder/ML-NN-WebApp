import streamlit as st
import pandas as pd

st.title(" ARTIFICIAL INTELLIGENCE PROJECT ")
tabs = st.tabs(["ME","MACHINE LEARNING", "NEURAL NETWORKS"])

with tabs[0]:
    st.subheader("👨‍💻 แนะนำตัวเอง")
    
    # ข้อมูลแนะนำตัว
    st.markdown("""
    **สวัสดีครับ 👋**  
    ผมชื่อ **สิทธิพงษ์ วงศ์สุวรรณ**  
    รหัสนักศึกษา **6404062636544**  
    กำลังศึกษาอยู่ที่ **มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ**  
    โครงงานนี้เกี่ยวข้องกับ **การพัฒนาโมเดล AI และการเทรน MODEL**  
    """)

    # เพิ่มรูปภาพ (Optional)
    st.image("Image/me.jpg", width=200, caption="Sittipong Wongsuwan")

    # ข้อมูลการติดต่อ (Optional)
    st.markdown("""
    🖥 **GitHub:** [https://github.com/sittipongisfolder](https://github.com/sittipongisfolder)  
   
    """)


with tabs[1]:

    st.header("📌 คำอธิบายของ Dataset")

    st.markdown("""
    ✅ **Zoo Animal Classification Dataset** เป็นชุดข้อมูลที่ใช้สำหรับ **การจำแนกประเภทของสัตว์**  
    ✅ มีข้อมูลเกี่ยวกับ **ลักษณะทางกายภาพของสัตว์** เช่น มีขน, มีขนนก, มีพิษ, หายใจด้วยปอด หรืออาศัยในน้ำ  
    ✅ **มีทั้งหมด 101 รายการของสัตว์** และ **7 หมวดหมู่หลัก (Class Types)**  
    """)

    # ----------------- ตารางแสดงหมวดหมู่ของสัตว์ ----------------- #
    st.subheader("🔹 หมวดหมู่ของสัตว์ (Class Type)")

    # สร้าง DataFrame สำหรับแสดงหมวดหมู่สัตว์
    class_data = {
        "ค่า (class_type)": [1, 2, 3, 4, 5, 6, 7],
        "ประเภทของสัตว์": [
            "Mammals (สัตว์เลี้ยงลูกด้วยนม)",
            "Birds (นก)",
            "Reptiles (สัตว์เลื้อยคลาน)",
            "Fish (ปลา)",
            "Amphibians (สัตว์สะเทินน้ำสะเทินบก)",
            "Insects (แมลง)",
            "Mollusks (หอย, สัตว์ไม่มีกระดูกสันหลัง)"
        ]
    }

    df_classes = pd.DataFrame(class_data)

    # แสดงตาราง
    st.table(df_classes)

    # ----------------- การนำ Dataset ไปใช้ ----------------- #
    st.subheader("📌 การนำ Dataset ไปใช้")

    st.markdown("""
    ✅ **การจำแนกประเภทของสัตว์ (Classification)**  
    - ใช้ **Machine Learning Model** เช่น Decision Tree, Random Forest, CatBoost, XGBoost  
    - สามารถฝึกโมเดลเพื่อทำนายว่าสัตว์ที่กำหนดอยู่ในหมวดหมู่ไหน  

    ✅ **การศึกษาทางชีววิทยา**  
    - วิเคราะห์ว่าคุณลักษณะของสัตว์แต่ละชนิดมีอิทธิพลต่อการแบ่งประเภทสัตว์อย่างไร  
    """)

 
with tabs[2]:
    st.header("📌 คำอธิบายของ Dataset")

    st.markdown("""
    ✅ **Netflix Stock Price Dataset** เป็นชุดข้อมูลที่ใช้สำหรับ **การวิเคราะห์และพยากรณ์ราคาหุ้นของ Netflix (NFLX)**  
    ✅ มีข้อมูลเกี่ยวกับ **ราคาเปิด, ราคาปิด, ราคาสูงสุด, ราคาต่ำสุด และปริมาณการซื้อขาย**  
    ✅ เหมาะสำหรับ **การใช้ Neural Networks (LSTM) เพื่อพยากรณ์ราคาหุ้นในอนาคต**  
    """)

    # ----------------- ตารางแสดงหมวดหมู่ของข้อมูล ----------------- #
    st.subheader("🔹 หมวดหมู่ของข้อมูลที่ใช้วิเคราะห์")

    # สร้าง DataFrame สำหรับแสดงหมวดหมู่ข้อมูล
    column_data = {
        "คอลัมน์": ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
        "คำอธิบาย": [
            "วันที่ของข้อมูล",
            "ราคาหุ้นตอนเปิดตลาด",
            "ราคาหุ้นสูงสุดของวัน",
            "ราคาหุ้นต่ำสุดของวัน",
            "ราคาหุ้นตอนปิดตลาด",
            "ราคาปิดที่ปรับปรุงแล้ว",
            "ปริมาณการซื้อขาย"
        ]
    }

    df_columns = pd.DataFrame(column_data)

    # แสดงตาราง
    st.table(df_columns)

    # ----------------- การนำ Dataset ไปใช้ ----------------- #
    st.subheader("📌 การนำ Dataset ไปใช้ด้วย Neural Networks (LSTM)")

    st.markdown("""
    ✅ **การพยากรณ์ราคาหุ้นด้วย LSTM (Long Short-Term Memory)**  
    - **ใช้ข้อมูล `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`** เป็น Feature  
    - **ใช้ `Close` เป็น Target Variable** เพื่อพยากรณ์ราคาหุ้นวันถัดไป  
    - ใช้ **Neural Networks (LSTM)** ซึ่งเป็น **ประเภทของ Recurrent Neural Network (RNN)**  
    - สามารถ **จดจำแนวโน้มข้อมูลในอดีต** และใช้พยากรณ์ราคาหุ้นในอนาคตได้อย่างแม่นยำ  

    ✅ **โครงสร้างของโมเดล LSTM**  
    - ใช้ **3 ชั้นของ LSTM**  
    - ใช้ **Dropout Layer** เพื่อป้องกัน Overfitting  
    - ใช้ **Dense Layer** สำหรับพยากรณ์ค่าราคาหุ้น  
    """)


