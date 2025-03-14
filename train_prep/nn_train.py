# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- โหลดข้อมูล ----------------- #
file_path = "NFLX.csv"  # เปลี่ยนเป็นพาธไฟล์ที่ถูกต้อง
data = pd.read_csv(file_path)

# แสดงตัวอย่างข้อมูล
print(data.head())

# ----------------- พล็อตกราฟราคาหุ้น Netflix ----------------- #
fig = make_subplots(specs=[[{"secondary_y": False}]])
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'].rolling(window=14).mean(), name="Netflix"), secondary_y=False)
fig.update_layout(title_text="Netflix Stock Price Trend")
fig.update_xaxes(title_text="Year")
fig.update_yaxes(title_text="Prices", secondary_y=False)
fig.show()

# ----------------- แบ่งข้อมูล Train และ Test ----------------- #
n = len(data)
train_data = data[0:(n//10)*9]  # 90% เป็น Training Set
test_data = data[(n//10)*9:]    # 10% เป็น Testing Set

print(f"Train Data Size: {len(train_data)}")
print(f"Test Data Size: {len(test_data)}")

# ----------------- ปรับขนาดข้อมูลด้วย MinMaxScaler ----------------- #
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data['Open'].values.reshape(-1, 1))

# ----------------- สร้างชุดข้อมูลสำหรับการเทรน ----------------- #
prediction_days = 30  # ใช้ข้อมูลย้อนหลัง 30 วัน
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)-10):  
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x+10, 0])  # พยากรณ์ 10 วันข้างหน้า

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(f"x_train Shape: {x_train.shape}")
print(f"y_train Shape: {y_train.shape}")

# ----------------- ฟังก์ชันสร้างโมเดล LSTM ----------------- #
def LSTM_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output Layer

    return model

# ----------------- สร้างและเทรนโมเดล ----------------- #
model = LSTM_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights_best.keras', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=20, batch_size=32, callbacks=[checkpointer])

# ----------------- แสดงกราฟ Loss และ Accuracy ----------------- #
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title("Model Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.show()

# ----------------- ทดสอบโมเดลกับ Test Data ----------------- #
actual_prices = test_data['Open'].values
total_dataset = pd.concat((train_data['Open'], test_data['Open']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# ----------------- แสดงผลลัพธ์การพยากรณ์ ----------------- #
plt.figure(figsize=(10, 5))
plt.plot(actual_prices, color='black', label="Actual Price")
plt.plot(predicted_prices, color='green', label="Predicted Price (10 Days Ahead)")
plt.title("Netflix Stock Price Prediction")
plt.xlabel("Days in Test Period")
plt.ylabel("Price")
plt.legend()
plt.show()

# ----------------- พยากรณ์ราคาหุ้นในอนาคต ----------------- #
real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Predicted Price for Next Day: {prediction[0][0]}")

# ----------------- บันทึกโมเดล ----------------- #
model.save('netflix_model.keras')

# ----------------- โหลดโมเดลที่บันทึกไว้ ----------------- #
from tensorflow import keras
loaded_model = keras.models.load_model('netflix_model.keras')

print("Model saved and loaded successfully!")
