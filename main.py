import os
import sys
import util
import numpy as np
import keras

epochs = 1000
load_num = 400
batch_size = 40
num_classes = 20
input_shape = (224, 224, 3)
model_type = 'vgg11' # ['vgg11', 'vgg16', 'minist']
weight_path = 'modle-%s.h5' % model_type

#util.extract_video('data/validate/*.mp4', 5, input_shape)

print('building modle...')
model = util.build_modle(model_type=model_type, input_shape=input_shape, num_classes=num_classes)

if os.path.exists(weight_path):
    print('load model weight...')
    model.load_weights(weight_path)

if '--train' in sys.argv:
    print('trainning data...')
    util.train(model, 'data/image/*.*', epochs=epochs, input_shape=input_shape, num_classes=num_classes, batch_size=batch_size, load_num=load_num, weight_path=weight_path)

print('predict data...')
util.predict(model, 'data/validate/*.*', input_shape=input_shape, num_classes=num_classes)
