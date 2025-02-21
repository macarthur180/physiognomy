import cv2
import time
import os
from datetime import datetime

# 저장할 디렉터리 설정
save_dir = "/home/nano/project/pic/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 카메라 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("카메라가 켜졌습니다. 3초 후에 사진을 촬영합니다.")
time.sleep(3)  # 3초 대기

# 프레임 캡처
ret, frame = cap.read()
if ret:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간 기반 파일명
    filename = os.path.join(save_dir, f"{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"사진이 저장되었습니다: {filename}")
else:
    print("사진 캡처에 실패했습니다.")

# 카메라 해제
cap.release()
cv2.destroyAllWindows()
