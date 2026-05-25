import matplotlib.image as mpimg
from deepface import DeepFace
import matplotlib.pyplot as plt
from typing import List

img_path = 'test_assets/IMG_1103.jpeg'
img = mpimg.imread(img_path)

plt.imshow(img)
plt.show()

img = DeepFace.extract_faces(img_path)

plt.imshow(img[0]['face'])
plt.show()

backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8n', 'yolov8m',
    'yolov8l', 'yolov11n', 'yolov11s', 'yolov11m',
    'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m',
    'yolov12l', 'yunet', 'centerface',
]
detector = backends[0]
align = True

objs: List[dict] = DeepFace.analyze(
    img_path=img_path,
    actions=['age', 'gender', 'race', 'emotion'],
    detector_backend = detector,
    align = align
)

best_face = max(objs, key=lambda f: f['face_confidence'])
print({
    "age": best_face['age'],
    "emotion": best_face['dominant_emotion'],
    "gender": best_face['dominant_gender'],
    "race": best_face['dominant_race'],
})