import matplotlib.image as mpimg
from deepface import DeepFace
import matplotlib.pyplot as plt
from pprint import pprint
from typing import List
import pandas as pd

img1_path = 'test_assets/sad.jpg'
img2_path = 'test_assets/angry.jpg'

img1 = mpimg.imread(img1_path)
img2 = mpimg.imread(img2_path)

plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.show()

img1 = DeepFace.extract_faces(img1_path)
img2 = DeepFace.extract_faces(img2_path)

plt.subplot(1, 2, 1)
plt.imshow(img1[0]['face'])
plt.subplot(1, 2, 2)
plt.imshow(img2[0]['face'])
plt.show()

model_name = 'Facenet'

resp = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, model_name = model_name)
pprint(resp)

objs: List[dict] = DeepFace.analyze(
 img_path = img1_path, actions = ['age', 'gender', 'race', 'emotion']
)
df = pd.DataFrame(objs)
print(df)
print(df.describe())

objs: List[dict] = DeepFace.analyze(
 img_path = img2_path, actions = ['age', 'gender', 'race', 'emotion']
)
df = pd.DataFrame(objs)
print(df)
print(df.describe())