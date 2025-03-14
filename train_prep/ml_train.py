# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import joblib

# ----------------- โหลดข้อมูล ----------------- #
file_path = "zoo.csv"  # เปลี่ยนเป็นพาธไฟล์ที่ถูกต้อง
df = pd.read_csv(file_path)

# ลบคอลัมน์ชื่อสัตว์ (ไม่จำเป็นต่อโมเดล)
df = df.drop(['animal_name'], axis=1)

# แบ่งข้อมูลเป็น Features (X) และ Target (y)
y = df['class_type']
X = df.drop(['class_type'], axis=1)

# แบ่งข้อมูลเป็น Train 67% และ Test 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# ----------------- เทรนโมเดล Random Forest ----------------- #
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# ทำนายผลและตรวจสอบ Accuracy
y_pred_rfc = rfc.predict(X_test)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print(f'Random Forest model accuracy: {accuracy_rfc:.4f}')

# ----------------- เทรนโมเดล CatBoost ----------------- #
catboost = CatBoostClassifier(random_state=42, verbose=0)
catboost.fit(X_train, y_train)

# ทำนายผลและตรวจสอบ Accuracy
y_pred_catboost = catboost.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
print(f'CatBoost model accuracy: {accuracy_catboost:.4f}')

# ----------------- บันทึกโมเดลที่เทรนแล้ว ----------------- #
joblib.dump(catboost, "catboost_model.cbm")
joblib.dump(rfc, "random_forest_model.pkl")

print("✅ โมเดลถูกบันทึกเรียบร้อยแล้ว!")

# ----------------- โหลดโมเดลที่บันทึกไว้ ----------------- #
loaded_catboost = joblib.load("catboost_model.cbm")
loaded_rfc = joblib.load("random_forest_model.pkl")

# ทดสอบโมเดลที่โหลดกลับมา
y_pred_loaded = loaded_catboost.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f'✅ Accuracy ของโมเดลที่โหลดกลับมา: {accuracy_loaded:.4f}')
