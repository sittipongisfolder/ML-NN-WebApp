import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# โหลดโมเดล LSTM ที่บันทึกไว้
@st.cache_resource
def load_model():
    return keras.models.load_model("models/netflix_model.keras")

model = load_model()

# โหลดข้อมูล Netflix Stock
@st.cache_resource
def load_data():
    data = pd.read_csv("datasets/NFLX.csv")
    return data

data = load_data()

# ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
if len(data) < 30:
    st.error("❌ ข้อมูลมีน้อยเกินไป กรุณาใช้ข้อมูลที่มีอย่างน้อย 30 วัน!")
    st.stop()

# ตั้งค่าการพยากรณ์
prediction_days = 30  # ใช้ข้อมูล 30 วันก่อนหน้า
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data["Open"].values.reshape(-1, 1))

st.title("📈 พยากรณ์ราคาหุ้น Netflix ด้วย Neural Network")
st.write("กรอกข้อมูลเพื่อให้โมเดลพยากรณ์ราคาหุ้น Netflix 🏢💰")

# แสดงกราฟราคาหุ้นทั้งหมด
st.subheader("📊 กราฟราคาหุ้นทั้งหมด")
plt.figure(figsize=(10, 5))
plt.plot(data["Open"].values, label="Real stock price", color="blue")
plt.title("Netflix historical price 12/16/2015 ~ 12/16/2019 daily")
plt.xlabel("DATE")
plt.ylabel("PRICE (USD)")
plt.legend()
st.pyplot(plt)

# เลือกจำนวนวันข้างหน้าที่ต้องการพยากรณ์
selected_days = st.slider("เลือกจำนวนวันข้างหน้าที่ต้องการพยากรณ์:", min_value=1, max_value=30, value=10)

# เตรียมข้อมูลล่าสุดสำหรับการพยากรณ์
total_dataset = data["Open"]
model_inputs = total_dataset[len(total_dataset) - prediction_days:].values.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# พยากรณ์ล่วงหน้าเป็นจำนวนวันที่เลือก
future_predictions = []
input_sequence = list(model_inputs[-prediction_days:].flatten())  # ✅ ใช้ list ที่มีค่า scalar เท่านั้น

# พยากรณ์ราคาหุ้นตามจำนวนวันที่เลือก
for _ in range(selected_days):
    x_test = np.array(input_sequence[-prediction_days:]).reshape(1, prediction_days, 1)
    predicted_price = model.predict(x_test)

    # ✅ แปลง predicted_price จาก numpy array เป็น scalar
    predicted_price_real = scaler.inverse_transform(predicted_price)[0, 0]
    
    future_predictions.append(predicted_price_real)

    # ✅ เพิ่มค่าใหม่ใน input_sequence โดยใช้ .item() เพื่อดึง scalar
    input_sequence.append(predicted_price.item())

# ปุ่มทำนายราคาหุ้น
if st.button("🔮 ทำนายราคาหุ้น"):
    st.success(f"📊 ราคาหุ้นที่คาดการณ์ในอีก {selected_days} วันข้างหน้า: **{future_predictions[-1]:.2f} USD**")

    # แสดงกราฟเปรียบเทียบ
    st.subheader("📈 การเปรียบเทียบราคาหุ้นจริงกับค่าพยากรณ์")
    plt.figure(figsize=(10, 5))
    plt.plot(data["Open"].values, label="Real stock price", color="black")
    plt.axvline(len(data) - 1, color="red", linestyle="--", label="Forecast starting point")
    
    future_x = range(len(data), len(data) + selected_days)
    plt.plot(future_x, future_predictions, color="green", linestyle="dashed", label="Forecasted prices")
    
    plt.title("Netflix Stock Price Forecast")
    plt.xlabel("DATE")
    plt.ylabel("PRICE (USD)")
    plt.legend()
    st.pyplot(plt)
