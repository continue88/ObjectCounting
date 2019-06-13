
import util
import cv2
import numpy as np

epochs = 1000
epoch_save = 100
load_num = 200
batch_size = 40
num_classes = 20
input_shape = (256, 256, 3)
weight_path = 'modle.h5'

print('building modle...')
model = util.build_modle(input_shape, num_classes)

print('load model weight...')
model.load_weights(weight_path)

video = cv2.VideoCapture('data/test.mp4')
success, img = video.read()
while success:
    # 裁剪成正方形
    height, width, _ = img.shape
    size = min(height, width)
    x = int((width - size) / 2)
    y = int((height - size) / 2)
    img = img[y:y+size, x:x+size]
    # 转换到网络输入
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    data_input = np.array([img]).astype(np.float32) / 255.0
    # 开始预测
    result = model.predict(data_input)
    count = np.argmax(result[0])
    # 显示结果
    cv2.putText(img, 'count:%d'%count, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.imshow('image', img)
    if cv2.waitKey(50) == 0:
        break
    success, img = video.read()
video.release()