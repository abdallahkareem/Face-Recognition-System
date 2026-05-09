
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from mtcnn import MTCNN
import cv2

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# إنشاء detector
detector = MTCNN()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # تحويل BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # كشف الوجوه
    faces = detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face['box']

        # رسم المستطيل
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # landmarks
        keypoints = face['keypoints']

        for point in keypoints.values():
            cv2.circle(
                frame,
                point,
                2,
                (0, 0, 255),
                2
            )

    cv2.imshow("MTCNN Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()