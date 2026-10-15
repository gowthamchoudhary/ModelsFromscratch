
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
Gx = np.zeros_like(pixel_matrix)
Gy = np.zeros_like(pixel_matrix)

class HOG:
    def __init__(self,con=0.01):
        self.gray_matrix = None
        self.con = con
        self.Gx = None
        self.Gy = None

    def convert_img_matrix(self,gray_image):
        image = cv2.resize(gray_image,(800,600))

        pixel_matrix = np.array(image)
        self.gray_matrix = pixel_matrix
        return pixel_matrix
    def calculate_gradients(self):
        self.Gx = np.zeros_like(self.gray_matrix)
        self.Gy = np.zeros_like(self.gray_matrix)
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
        magnitude = np.sqrt(self.Gx**2 + self.Gy**2)
        angle = np.degrees(np.arctan2(self.Gy, self.Gx)) % 180
        
        


     




# cv2.imshow("Grayscale Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
