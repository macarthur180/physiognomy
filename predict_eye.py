import cv2
import dlib
import numpy as np
import joblib
import os

# 모델 파일 로드
model_path = "/home/nano/project/eye_model.pkl"  # 학습된 모델 경로
label_encoder_path = "/home/nano/project/label_encoder.pkl"  # 저장된 라벨 인코더 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/nano/project/shape_predictor_68_face_landmarks.dat")

# 저장된 사진 폴더 경로
pic_dir = "/home/nano/project/pic/"

# 가장 최근 촬영된 사진 찾기
def get_latest_image(directory):
    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    if not files:
        print("❌ 저장된 사진이 없습니다.")
        exit()
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)

latest_image = get_latest_image(pic_dir)
print(f"🔍 분석할 사진: {latest_image}")

# 이미지 로드
img = cv2.imread(latest_image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)
if len(faces) == 0:
    print("❌ 얼굴을 찾을 수 없습니다.")
    exit()

for face in faces:
    landmarks = predictor(gray, face)

    # 얼굴 크기
    face_width = face.right() - face.left()
    face_height = face.bottom() - face.top()

    # 왼쪽 눈 (36~41)
    left_eye_x = [landmarks.part(n).x for n in range(36, 42)]
    left_eye_y = [landmarks.part(n).y for n in range(36, 42)]
    left_eye_width = max(left_eye_x) - min(left_eye_x)
    left_eye_height = max(left_eye_y) - min(left_eye_y)

    # 오른쪽 눈 (42~47)
    right_eye_x = [landmarks.part(n).x for n in range(42, 48)]
    right_eye_y = [landmarks.part(n).y for n in range(42, 48)]
    right_eye_width = max(right_eye_x) - min(right_eye_x)
    right_eye_height = max(right_eye_y) - min(right_eye_y)

    # 눈 사이 거리 (39번과 42번 사이 거리)
    eye_distance = np.linalg.norm(
        np.array([landmarks.part(39).x, landmarks.part(39).y]) - 
        np.array([landmarks.part(42).x, landmarks.part(42).y])
    )

    # 눈꼬리 기울기 (39번과 42번 사이 기울기)
    eye_angle = np.arctan2(landmarks.part(42).y - landmarks.part(39).y, 
                            landmarks.part(42).x - landmarks.part(39).x) * 180 / np.pi

    # 비율 계산
    left_eye_ratio = left_eye_width / face_width
    right_eye_ratio = right_eye_width / face_width
    eye_distance_ratio = eye_distance / face_width
    eye_height_ratio = (left_eye_height + right_eye_height) / (2 * face_height)

    # 모델 및 라벨 인코더 로드
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # 예측 수행
    features = np.array([[left_eye_ratio, right_eye_ratio, eye_distance_ratio, eye_height_ratio, eye_angle]])
    prediction = model.predict(features)

    # 숫자를 원래 눈 유형으로 변환
    predicted_eye_type = label_encoder.inverse_transform(prediction)

    print(f"🔮 예측된 눈 유형: {predicted_eye_type[0]}")  # 예측된 관상 유형 출력
