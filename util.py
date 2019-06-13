
import cv2
import glob
import numpy as np

def extract_video(path, skip, input_shape):
    file_list = glob.glob(path)
    height, width, channels = input_shape
    for file in file_list:
        vidcap = cv2.VideoCapture(file)
        sucess, img = vidcap.read()
        frame = 0
        while sucess:
            #cv2.putText(img, file, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            img = cv2.resize(img, (width, height))
            if frame % skip == 0:
                img_file = file.replace('.mp4', '_%d.png' % frame)
                cv2.imwrite(img_file, img)
            frame += 1
            print('processing %s %d' % (file, frame))
            sucess, img = vidcap.read()
        vidcap.release()

def rotate_image(image, angle):
    row, col, channel = image.shape
    center = (row / 2, col / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_img = cv2.warpAffine(image, rot_mat, (col, row), cv2.INTER_LINEAR, 0, cv2.BORDER_REPLICATE)
    return new_img

def load_dataset(path, load_num, input_shape, train=True):
    file_list = glob.glob(path)
    if load_num < len(file_list):
        file_list = np.random.choice(file_list, load_num, replace=False)
    height, width, channels = input_shape
    img_list = []
    label_list = []
    rotate_range = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for file in file_list:
        begin = file.rfind('\\') + 1
        end = file.rfind(' ')
        if end < 0:
            end = file.rfind('.')
        label = int(file[begin:end])
        img = cv2.imread(file)
        img = cv2.resize(img, (width, height))
        if train:
            rot = np.random.randint(0, len(rotate_range))
            angle = rotate_range[rot]
            img = rotate_image(img, angle)
        img_list.append(img)
        label_list.append(label)
    return (img_list, label_list)
