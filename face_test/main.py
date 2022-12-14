from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("train/e1.jpeg")
print(img1)
plt.imshow(img1[:,:,::-1])
plt.show()

img2 = cv2.imread("test/e2.jpg")
print(img2)
plt.imshow(img2[:,:,::-1])
plt.show()

result = DeepFace.verify(img1, img2)
print(result)