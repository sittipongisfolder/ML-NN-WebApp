# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import joblib

# 📂 กำหนด Path ของ Dataset
file_path = "datasets/zoo.csv"  # ปรับให้เป็น path ที่ถูกต้อง
if not os.path.exists(file_path):
    raise FileNotFoundError(f"🚨 ไม่พบไฟล์ข้อมูล: {file_path}")

# 📊 โหลดข้อมูล
df = pd.read_csv(file_path)

# 🔹 ลบคอลัมน์ animal_name เพราะไม่มีผลกับการพยากรณ์
df = df.drop(columns=["animal_name"])

# 🚀 แยก Features และ Target
y = df["class_type"]
X = df.drop(columns=["class_type"])

# 🔹 แบ่งข้อมูลเป็น Train (67%) และ Test (33%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# ----------------- 🚀 เทรนโมเดล Random Forest ----------------- #
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# ✅ ทำนายผลและคำนวณ Accuracy
y_pred_rfc = rfc.predict(X_test)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print(f"🌲 Random Forest Accuracy: {accuracy_rfc:.4f}")

# ----------------- 🚀 เทรนโมเดล CatBoost ----------------- #
catboost_model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function='MultiClass', verbose=50, random_seed=42)
catboost_model.fit(X_train, y_train)

# ✅ ทำนายผลและคำนวณ Accuracy
y_pred_catboost = catboost_model.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
print(f"🐱 CatBoost Accuracy: {accuracy_catboost:.4f}")

# 📂 สร้างโฟลเดอร์ models ถ้ายังไม่มี
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# ----------------- 💾 บันทึกโมเดล ----------------- #
catboost_path = os.path.join(model_dir, "catboost_model.cbm")
random_forest_path = os.path.join(model_dir, "random_forest_model.pkl")

# ✅ บันทึก CatBoost ใช้ save_model()
catboost_model.save_model(catboost_path, format="cbm")
print(f"✅ โมเดล CatBoost ถูกบันทึกที่: {catboost_path}")

# ✅ บันทึก Random Forest ใช้ joblib
joblib.dump(rfc, random_forest_path)
print(f"✅ โมเดล Random Forest ถูกบันทึกที่: {random_forest_path}")

# ----------------- 🔍 ทดสอบโหลดโมเดล ----------------- #
loaded_catboost = CatBoostClassifier()
loaded_catboost.load_model(catboost_path)

loaded_rfc = joblib.load(random_forest_path)

# ✅ ทดสอบโมเดลที่โหลดกลับมา
y_pred_loaded = loaded_catboost.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"🎯 Accuracy ของ CatBoost ที่โหลดกลับมา: {accuracy_loaded:.4f}")
