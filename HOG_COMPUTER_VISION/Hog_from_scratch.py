
import cv2
import numpy as np

img = cv2.imread(
    r"C:\From Scratch\ModelsFromscratch\HOG_COMPUTER_VISION\test-1.png"
)

if img is None:
    raise FileNotFoundError("No image is found")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.resize(gray,(800,600))

pixel_matrix = np.array(image)


cv2.imshow("Grayscale Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
