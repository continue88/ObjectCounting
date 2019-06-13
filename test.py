
import util
import cv2


image = cv2.imread('data/image/19 (18).JPG')
image = cv2.resize(image, (512, 512))
rote_image = util.rotate_image(image, 30)
while cv2.waitKey(50) != 0:
    cv2.imshow('image', rote_image)
