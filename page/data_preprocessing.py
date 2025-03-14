import streamlit as st
import pandas as pd

st.markdown('<h2 style="color:blue;">PROCESS OF DEVELOPING MACHINE LEARNING AND NEURAL NETWORK MODELS</h2>', unsafe_allow_html=True)           

tabs = st.tabs(["Preparing data ML","Preparing data NN", "Theory of Developed Algorithms ML","Theory of Developed Algorithms NN", "ML Dev process", "NN Dev process"])


with tabs[0]:
    
    st.title("Preparation")
    st.subheader("🔍 1. รายละเอียดของข้อมูล")
    
    st.markdown("""
    - **📄 จำนวนข้อมูล**: 101 แถว และ 18 คอลัมน์  
    - **📝 คอลัมน์สำคัญ**:
        - `animal_name`: ชื่อของสัตว์ (ประเภทข้อความ)
        - คุณลักษณะของสัตว์ (Feature Columns):  
            - 🦁 `hair`, `feathers`, `eggs`, `milk`, `airborne`, `aquatic`
            - 🐍 `predator`, `toothed`, `backbone`, `breathes`, `venomous`, `fins`
            - 🐾 `legs`, `tail`, `domestic`, `catsize`
        - `class_type`: หมวดหมู่ของสัตว์ (label)
    - **📊 รูปแบบข้อมูล**:
        - ตัวเลข (`0 = ไม่มี`, `1 = มี`)
        - `legs` เป็นค่าตัวเลข (0, 2, 4, 6, 8)
        - `class_type` เป็น Label ที่ใช้แบ่งประเภทของสัตว์
    """)

    st.subheader("🛠 2. การเตรียมข้อมูล")

    st.markdown("""
    ### (a) ตรวจสอบค่าหายไป (Missing Values)
    - ✅ จาก `df.info()` พบว่า **ไม่มีค่าหายไป** (ทุกคอลัมน์มีข้อมูลครบ)

    ### (b) การแปลงค่าข้อมูล
    - 🔢 **ข้อมูลส่วนใหญ่เป็นตัวเลข** (`int64`) ✅ ไม่ต้องแปลงค่า
    - 🏷️ `animal_name` เป็นข้อความ **อาจไม่นำมาใช้** ในการวิเคราะห์เชิงตัวเลข

    ### (c) การจัดการคุณลักษณะ (Feature Engineering)
    - 🛑 พิจารณา **ลบคอลัมน์ `animal_name`** หากไม่ได้ใช้
    - 🎯 ใช้ `class_type` เป็น **Target Variable**
    - 🔄 ทำให้ `legs` เป็น **ค่าหมวดหมู่** (0, 2, 4, 6, 8)
    - 🔢 ใช้ **One-Hot Encoding** แปลงค่าหมวดหมู่เป็นตัวเลข

    ### (d) การปรับสมดุลข้อมูล (Data Balancing)
    - 📊 ตรวจสอบการกระจายตัวของ `class_type`
    - ⚖️ หากข้อมูลไม่สมดุล อาจใช้ **Oversampling / Undersampling**
    """)

    st.subheader("📌 3. สรุปการเตรียมข้อมูล")

    st.markdown("""
    ✅ **ลบคอลัมน์ `animal_name`** หากไม่ได้ใช้  
    ✅ **ตรวจสอบค่าหายไป** (ไม่มีค่าหายไปใน dataset นี้)  
    ✅ **ตรวจสอบค่าผิดปกติ** เช่น `legs` ที่อาจมีค่าผิดปกติ  
    ✅ **ปรับค่า `legs` เป็นค่าหมวดหมู่** และใช้ **One-Hot Encoding**  
    ✅ **เช็คความสมดุลของ `class_type`** และใช้เทคนิค Balancing หากจำเป็น  
    """)
with tabs[1]:
    st.title("Preparation")
    st.subheader("🔍 1. รายละเอียดของข้อมูล")
    
    st.markdown("""
    - **📄 จำนวนข้อมูล**: 1007 แถว และ 7 คอลัมน์  
    - **📊 คอลัมน์สำคัญ**:
        - `Date`: วันที่ของข้อมูล
        - `Open`, `High`, `Low`, `Close`, `Adj Close`: ราคาหุ้นในแต่ละวัน
        - `Volume`: ปริมาณการซื้อขาย
    - **🔎 สังเกตข้อมูล**:
        - `Date` เป็น **object** ต้องแปลงเป็น **datetime**
        - ไม่มี **ค่าหายไป** ✅
    """)

    st.subheader("🛠 2. การเตรียมข้อมูล")

    st.markdown("""
    ### (a) ตรวจสอบค่าหายไป (Missing Values)
    - ✅ ไม่มีค่าหายไป ทุกคอลัมน์มีข้อมูลครบ

    ### (b) การแปลงค่าข้อมูล
    - 🔄 แปลง `Date` เป็น **datetime**
    
    ### (c) การจัดการคุณลักษณะ (Feature Engineering)
    - 🏷 **เพิ่มฟีเจอร์วันเวลา** เช่น Year, Month, Day, Day of Week
    - 📊 **สร้างค่าเฉลี่ยเคลื่อนที่ (Moving Average)**
        - MA_7 (ค่าเฉลี่ย 7 วัน)
        - MA_30 (ค่าเฉลี่ย 30 วัน)
    - 📉 **สร้างฟีเจอร์การเปลี่ยนแปลงของราคา** เช่น `Daily Change (%)`

    ### (d) การตรวจสอบแนวโน้มข้อมูล (Data Trends)
    - 📈 พล็อตแนวโน้มราคาหุ้น
    - 🔄 ตรวจสอบความสัมพันธ์ระหว่าง `Volume` กับ `Price Movement`
    """)

    st.subheader("📌 3. สรุปการเตรียมข้อมูล")

    st.markdown("""
    ✅ **แปลง `Date` เป็น datetime**  
    ✅ **เพิ่มฟีเจอร์ Year, Month, Day, Day of Week**  
    ✅ **สร้างฟีเจอร์ Moving Average (MA_7, MA_30)**  
    ✅ **คำนวณ Daily Change (%)**  
    ✅ **ตรวจสอบแนวโน้มราคาและปริมาณการซื้อขาย**  
    """)


with tabs[2]:
    st.title("🧠 ทฤษฎีของอัลกอริทึมที่พัฒนา")

    # ----------------- จำนวนอัลกอริทึมที่ใช้ ----------------- #
    st.header("📌 อัลกอริทึมที่ใช้ในการเทรนโมเดล")
    st.markdown("""
    ในการเทรนโมเดลจากข้อมูล **zoo.csv** ใช้อัลกอริทึม **2 ตัวหลัก** ได้แก่:

    1️⃣ **Random Forest Classifier**  
    2️⃣ **CatBoost Classifier**  

    แต่ละอัลกอริทึมมีข้อดีดังนี้:
    """)

    # ----------------- ทฤษฎีของอัลกอริทึม ----------------- #
    st.subheader("🌲 1. Random Forest Classifier")

    st.markdown("""
    ✅ **เป็นอัลกอริทึมแบบ Ensemble Learning** ที่รวมหลาย **Decision Trees**  
    ✅ **ทำงานโดยการสุ่มตัวอย่างข้อมูล** และสร้างหลายต้นไม้เพื่อหาผลลัพธ์ที่ดีที่สุด  
    ✅ **แข็งแกร่งต่อข้อมูลที่มีเสียงรบกวน (Noise) และ Overfitting น้อยกว่า Decision Tree ปกติ**  
    ✅ **เหมาะกับปัญหา Classification และ Regression**  
    """)

    st.subheader("🚀 2. CatBoost Classifier")

    st.markdown("""
    ✅ **เป็นอัลกอริทึม Gradient Boosting ที่ออกแบบมาสำหรับข้อมูลที่มีฟีเจอร์แบบ Categorical**  
    ✅ **สามารถเรียนรู้ข้อมูลได้ดีโดยไม่ต้องทำ One-Hot Encoding**  
    ✅ **ทำงานได้เร็วและแม่นยำกว่าการ Boosting ทั่วไป เช่น XGBoost และ LightGBM**  
    ✅ **ใช้ได้ดีกับข้อมูลขนาดเล็กถึงขนาดใหญ่**  
    """)

    # ----------------- กระบวนการทำงาน ----------------- #
    st.header("🛠 กระบวนการทำงานของโมเดล")

    st.markdown("""
    1️⃣ **โหลดข้อมูลและเตรียมข้อมูล**  
    - ลบคอลัมน์ `animal_name` เพราะไม่เกี่ยวข้องกับการเทรน  
    - แบ่งข้อมูลเป็น `X` (ฟีเจอร์) และ `y` (label)  
    - แบ่งข้อมูลเป็น Train 67% และ Test 33%  

    2️⃣ **เทรนโมเดลด้วย Random Forest**  
    - ใช้ค่า **Random State = 42** เพื่อให้ได้ผลลัพธ์เดิมทุกครั้ง  
    - ใช้ **accuracy_score** เช็คความแม่นยำของโมเดล  

    3️⃣ **เทรนโมเดลด้วย CatBoost**  
    - ใช้ค่า **Random State = 42** และเปิด `verbose=0` เพื่อให้โมเดลเทรนแบบเงียบ  
    - บันทึกโมเดลเป็นไฟล์ **catboost_model.cbm**  

    4️⃣ **บันทึกโมเดลและเตรียมใช้ในอนาคต**  
    - ใช้ `joblib.dump()` บันทึกโมเดล  
    - ใช้ `joblib.load()` โหลดกลับมาใช้งาน  
    """)


with tabs[3]:
    st.title("ทฤษฎีของอัลกอริทึมที่ใช้พัฒนา")
    st.markdown("""
        ### 🔹 **1. LSTM คืออะไร?**
        LSTM (Long Short-Term Memory) เป็นอัลกอริทึมประเภท **Recurrent Neural Network (RNN)**  
        ที่ถูกออกแบบมาให้สามารถ **จดจำข้อมูลระยะยาว** ได้ดีกว่า RNN ปกติ  
        เหมาะสำหรับ **การพยากรณ์ข้อมูลที่เป็นลำดับเวลา (Time Series Prediction)** เช่น:
        - การพยากรณ์ราคาหุ้น
        - การวิเคราะห์แนวโน้มอุณหภูมิ
        - การรู้จำเสียงพูด หรือข้อความ

        ### 🔹 **2. หลักการทำงานของ LSTM**
        LSTM ใช้ **เซลล์หน่วยความจำ (Memory Cells)** ที่ช่วยให้โมเดลสามารถเลือก **จดจำ** หรือ **ลืม** ข้อมูลที่สำคัญได้  
        โดยมี **3 ประตูหลัก** (Gates) ทำงานร่วมกัน:
        1. **Forget Gate** – ตัดสินใจว่าจะลืมข้อมูลเก่าหรือไม่  
        2. **Input Gate** – เลือกข้อมูลใหม่ที่จะบันทึกลงหน่วยความจำ  
        3. **Output Gate** – กำหนดค่าผลลัพธ์ที่ต้องส่งต่อไปยังเซลล์ถัดไป  

        ### 🔹 **3. ทำไมต้องใช้ LSTM แทน RNN ธรรมดา?**
        ✅ **RNN ปกติ** มีปัญหา **Gradient Vanishing** เมื่อข้อมูลมีลำดับยาวเกินไป  
        ✅ **LSTM** แก้ไขปัญหานี้โดยมี **Memory Cell** ที่ช่วยเก็บข้อมูลที่สำคัญได้นานขึ้น  
        ✅ **เหมาะกับข้อมูลที่ต้องจดจำบริบทระยะยาว** เช่น ราคาหุ้น หรือคำในประโยค  

        ### 🔹 **4. LSTM ในการพยากรณ์ราคาหุ้น**
        อัลกอริทึมนี้ใช้ **ข้อมูลราคาหุ้นย้อนหลัง 30 วัน** เพื่อทำนายราคาหุ้นใน **10 วันถัดไป**  
        โดยโมเดลจะเรียนรู้แนวโน้มของตลาดจากข้อมูลที่ผ่านมา แล้วพยากรณ์ค่าใหม่ให้แม่นยำที่สุด  

        ### 🔹 **5. กระบวนการทำงานของโมเดล**
        1️⃣ **โหลดและทำความสะอาดข้อมูล** – ตรวจสอบค่าหายไป และปรับรูปแบบวันที่  
        2️⃣ **แปลงข้อมูลเป็น Normalization** – ใช้ **MinMaxScaler** เพื่อให้ข้อมูลอยู่ในช่วง 0-1  
        3️⃣ **สร้างชุดข้อมูล Train และ Test** – แบ่งข้อมูล 90% สำหรับเทรน และ 10% สำหรับทดสอบ  
        4️⃣ **สร้างโมเดล LSTM** – ใช้ชั้น LSTM 3 ชั้น พร้อม **Dropout** ป้องกัน Overfitting  
        5️⃣ **เทรนโมเดล** – ปรับค่าการเรียนรู้และลดค่าความผิดพลาด (Loss)  
        6️⃣ **ทดสอบโมเดล** – ใช้ข้อมูล Test เพื่อทำนายราคาหุ้นและเปรียบเทียบกับราคาจริง  
        7️⃣ **บันทึกโมเดล** – เซฟโมเดลที่ดีที่สุดเพื่อใช้งานต่อไป  

        ### 🔹 **6. จุดเด่นของโมเดลที่พัฒนา**
        ✅ **ใช้ LSTM Neural Network** เพื่อพยากรณ์ราคาหุ้น  
        ✅ **ใช้ข้อมูลย้อนหลัง 30 วัน** เพื่อทำนายราคาหุ้นใน **10 วันข้างหน้า**  
        ✅ **ใช้เทคนิคการปรับค่า (Normalization & Feature Engineering)** เพื่อเพิ่มความแม่นยำ  
        ✅ **สามารถบันทึกและโหลดโมเดลที่ฝึกมาแล้ว** เพื่อใช้งานใหม่ได้  "
        """)

with tabs[4]:
    st.header("📌 1. Machine Learning (ML) Model Development")

    st.markdown("""
        ### 🔹 **1️⃣ ขั้นตอนการพัฒนาโมเดล Machine Learning**
        การพัฒนาโมเดล Machine Learning (ML) ผ่าน **6 ขั้นตอนหลัก** ดังนี้:

        1️⃣ **เตรียมข้อมูล (Data Preparation)**  
        - โหลดและทำความสะอาดข้อมูล (ลบค่า Missing, ลบ Outlier)  
        - แปลงข้อมูล Categorical เป็นตัวเลข (One-Hot Encoding)  
        - ปรับสเกลข้อมูลให้เหมาะสม (Normalization / Standardization)  

        2️⃣ **แบ่งข้อมูลเป็นชุด Train และ Test (Train-Test Split)**  
        - ใช้ `train_test_split()` แบ่งข้อมูลออกเป็น **Train 80% - Test 20%**  
        - ใช้ `X_train, X_test, y_train, y_test` เพื่อแยกฟีเจอร์และ label  

        3️⃣ **เลือกอัลกอริทึมที่เหมาะสม (Model Selection)**  
        - 🎯 **Random Forest** – เหมาะกับ Classification และ Regression  
        - 🚀 **CatBoost** – เหมาะกับข้อมูลที่มี Categorical Features  

        4️⃣ **เทรนโมเดล (Training the Model)**  
        - ใช้ `model.fit(X_train, y_train)` เทรนโมเดล  
        - ใช้ `cross-validation` เพื่อเลือกพารามิเตอร์ที่ดีที่สุด  

        5️⃣ **ทดสอบและปรับแต่งโมเดล (Evaluation & Tuning)**  
        - ใช้ `accuracy_score()` หรือ `mean_squared_error()` ตรวจสอบผลลัพธ์  
        - ปรับค่าพารามิเตอร์ (Hyperparameter Tuning)  

        6️⃣ **บันทึกโมเดลที่ดีที่สุด (Save the Model)**  
        - ใช้ `joblib.dump(model, 'catboost_model.cbm")` บันทึกโมเดล  
        - ใช้ `joblib.load("'catboost_model.cbm")` โหลดกลับมาใช้งาน  

        ✅ **สรุป**: โมเดล ML จะใช้การเรียนรู้แบบ Supervised Learning ที่อาศัยข้อมูลตัวอย่างในการสอนโมเดล 🚀  
        """)

    data = {
        "ขั้นตอน": [
            "1️⃣ เตรียมข้อมูล (Data Preparation)",
            "2️⃣ แบ่งข้อมูล (Train-Test Split)",
            "3️⃣ เลือกอัลกอริทึม (Model Selection)",
            "4️⃣ เทรนโมเดล (Training the Model)",
            "5️⃣ ประเมินผล (Evaluation & Tuning)",
            "6️⃣ บันทึกและนำไปใช้งานจริง (Save & Deploy)"
        ],
        "รายละเอียด": [
            "โหลดและทำความสะอาดข้อมูล, ตรวจสอบค่า Missing, จัดการ Outliers",
            "แบ่งข้อมูลเป็น Train 80% และ Test 20%",
            "เลือกโมเดล เช่น Random Forest, CatBoost",
            "ฝึกโมเดลโดยใช้ catboost.fit(X_train, y_train)",
            "ใช้ Accuracy, MSE วัดผล และปรับค่าพารามิเตอร์",
            "บันทึกโมเดลด้วย jcatboost และนำไปใช้กับ Web App หรือ API"
        ]
    }

    df = pd.DataFrame(data)

# แสดงตารางใน Streamlit
    st.table(df)


with tabs[5]:
    st.header("📌 2. Neural Network (NN) Model Development")

    st.markdown("""
    ### 🔹 **2️⃣ ขั้นตอนการพัฒนาโมเดล Neural Network (Deep Learning)**
    โมเดล Neural Network มีขั้นตอนพัฒนา **7 ขั้นตอนหลัก**:

    1️⃣ **เตรียมข้อมูล (Data Preparation)**
    - ใช้ **MinMaxScaler** หรือ **StandardScaler** ปรับค่าข้อมูล  
    - แปลงข้อมูล Categorical เป็น One-Hot Encoding  
    - แบ่งข้อมูลเป็น Train/Test เช่นเดียวกับ Machine Learning  

    2️⃣ **สร้างโครงสร้างของโมเดล (Model Architecture)**
    - ใช้ `Sequential()` สร้างโมเดล  
    - กำหนดจำนวน **Layers และ Neurons**  
    - ใช้ Activation Functions เช่น ReLU, Sigmoid, Softmax  

    3️⃣ **เลือก Optimizer และ Loss Function**
    - Optimizer: `Adam`, `SGD`, `RMSprop`  
    - Loss Function: `categorical_crossentropy`, `mse`  

    4️⃣ **เทรนโมเดล (Training the Model)**
    - ใช้ `model.fit(X_train, y_train, epochs=50, batch_size=32)`  
    - ใช้ `EarlyStopping` และ `ReduceLROnPlateau` ป้องกัน Overfitting  

    5️⃣ **ตรวจสอบผลลัพธ์ (Evaluation)**
    - ใช้ `model.evaluate(X_test, y_test)` ตรวจสอบความแม่นยำ  
    - พล็อตกราฟ `loss` และ `accuracy` เพื่อตรวจสอบการเรียนรู้  

    6️⃣ **บันทึกและโหลดโมเดล (Save & Load Model)**
    - ใช้ `model.save("model.h5")` บันทึกโมเดล  
    - ใช้ `keras.models.load_model("model.h5")` โหลดกลับมาใช้  

    7️⃣ **นำไปใช้งานจริง (Deployment)**
    - ใช้โมเดลกับ **Web App / API** เพื่อให้ผู้ใช้สามารถทดสอบโมเดล  
    - ปรับแต่งโมเดลให้รองรับข้อมูลที่เปลี่ยนแปลงในอนาคต  

    ✅ **สรุป**: Neural Network เหมาะกับงานที่ซับซ้อน เช่น **การพยากรณ์ข้อมูล, ภาพ, ข้อความ และเสียง**  
    """)
    data = {
        "ขั้นตอน": [
            "1️⃣ เตรียมข้อมูล (Data Preparation)",
            "2️⃣ ออกแบบโครงสร้างโมเดล (Model Architecture)",
            "3️⃣ เลือกฟังก์ชัน Loss และ Optimizer",
            "4️⃣ เทรนโมเดล (Training the Model)",
            "5️⃣ ประเมินผลและปรับแต่ง (Evaluation & Fine-Tuning)",
            "6️⃣ บันทึกและโหลดโมเดล (Save & Load Model)",
            "7️⃣ นำไปใช้งานจริง (Deployment)"
        ],
        "รายละเอียด": [
            "โหลดข้อมูล, ทำความสะอาด, ปรับค่าให้เหมาะสม (Normalization, One-Hot Encoding)",
            "กำหนดจำนวนชั้นของ Neural Network และเลือก Activation Functions (ReLU, Sigmoid, Softmax)",
            "ใช้ Loss Function เช่น CrossEntropy หรือ MSE และ Optimizer เช่น Adam, SGD",
            "ฝึกโมเดลโดยใช้ model.fit(X_train, y_train, epochs=50, batch_size=32)",
            "ใช้ Accuracy, Loss ตรวจสอบโมเดล และปรับค่าพารามิเตอร์ (Hyperparameter Tuning)",
            "บันทึกโมเดลด้วย .h5 หรือ .keras และโหลดกลับมาใช้งาน",
            "นำโมเดลไปใช้ใน Web App หรือ Mobile App ผ่าน API"
        ]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True)

    


