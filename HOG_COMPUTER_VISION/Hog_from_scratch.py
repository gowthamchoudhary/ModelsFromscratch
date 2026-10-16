
import cv2
import numpy as np


class HOG:
    def __init__(self,con=0.01):
        self.gray_matrix = None
        self.con = con
        self.Gx = None
        self.Gy = None
        self.magnitude = None
        self.angle = None
    def convert_to_gray(self,image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    def convert_img_matrix(self,gray_image):
        
        image = cv2.resize(gray_image,(800,600))

        pixel_matrix = np.array(image)
        self.gray_matrix = pixel_matrix
        return pixel_matrix
    def calculate_gradients(self):
        self.Gx = np.zeros_like(self.gray_matrix, dtype=np.float32)
        self.Gy = np.zeros_like(self.gray_matrix, dtype=np.float32)
        self.Gx[:, 1:-1] = (
        self.gray_matrix[:, 2:] - self.gray_matrix[:, :-2]
        ) / 2
        self.Gy[1:-1, :] = (
    self.gray_matrix[2:, :] - self.gray_matrix[:-2, :]
) / 2
        self.Gx[[0, -1], :] = 0
        self.Gx[:, [0, -1]] = 0

        self.Gy[[0, -1], :] = 0
        self.Gy[:, [0, -1]] = 0
        self.magnitude = np.sqrt(self.Gx**2 + self.Gy**2)
        self.angle = np.degrees(np.arctan2(self.Gy, self.Gx)) % 180
        return self.magnitude, self.angle

img = cv2.imread(
    r"C:\From Scratch\ModelsFromscratch\HOG_COMPUTER_VISION\test-1.png"
)

if img is None:
    raise FileNotFoundError("No image is found")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hog = HOG()
hog.convert_img_matrix(gray_image=gray)
mag,ang = hog.calculate_gradients()
display_image = cv2.normalize(
    mag, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)



vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print("cannot open the camera")
    exit()
while True:
    ret,frame = vid.read()
    if not ret:
        print("it is not streaming a shit bro !!!")
    gray_frame = hog.convert_to_gray(image=frame)
    gray_matrix = hog.convert_img_matrix(gray_frame)
    mag,ang = hog.calculate_gradients()
    display_frame = cv2.normalize(
        mag,None,0,255,cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imshow("streaming",display_frame)
    if cv2.waitKey(1)==ord('q'):
        break
vid.release()
cv2.destroyAllWindows()




cv2.imshow("Grayscale Image", gray)
cv2.imshow("Gradient Magnitude", display_image)

cv2.waitKey(0)
cv2.destroyAllWindows()