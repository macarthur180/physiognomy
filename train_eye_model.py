import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# CSV 데이터 불러오기
csv_path = "/home/nano/project/eye_data.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 입력값(X)과 라벨(y) 분리
X = df[["left_eye_ratio", "right_eye_ratio", "eye_distance_ratio", "eye_height_ratio", "eye_angle"]]
y = df["eye_type"]

# 라벨을 숫자로 변환
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 데이터를 학습 데이터(80%)와 테스트 데이터(20%)로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# SVM 모델 학습
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Random Forest 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 테스트 데이터 예측
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# 정확도 측정
svm_accuracy = accuracy_score(y_test, svm_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"✅ SVM 정확도: {svm_accuracy:.4f}")
print(f"✅ Random Forest 정확도: {rf_accuracy:.4f}")

# 더 높은 정확도의 모델을 선택하여 저장
if svm_accuracy > rf_accuracy:
    best_model = svm_model
    best_model_name = "SVM"
    best_model_path = "/home/nano/project/svm_eye_model.pkl"
else:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_model_path = "/home/nano/project/rf_eye_model.pkl"

print(f"🎯 선택된 최적 모델: {best_model_name}")

# 최적의 모델을 eye_model.pkl로 저장
joblib.dump(best_model, "/home/nano/project/eye_model.pkl")

# 라벨 인코더도 저장 (예측 시 필요)
joblib.dump(label_encoder, "/home/nano/project/label_encoder.pkl")

print("✅ 최적의 모델이 eye_model.pkl로 저장되었습니다!")
