
import util
import cv2
import numpy as np

num_classes = 20
input_shape = (256, 256, 3)
model_type = 'default' # ['vgg11', 'vgg16', 'minist', 'default']
weight_path = 'modle-%s.h5' % model_type
video_path = 'data/video.mp4'

print('building modle...')
model = util.build_modle(input_shape, num_classes, model_type=model_type)

print('load model weight...')
model.load_weights(weight_path)

video = cv2.VideoCapture(video_path)
success, img = video.read()
while success:
    # 裁剪倒正方型
    height, width, _ = img.shape
    if height != width:
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
    util.draw_text(img, 'count:%d'%count)
    cv2.imshow('image', img)
    if cv2.waitKey(50) == 0:
        break
    success, img = video.read()
video.release()