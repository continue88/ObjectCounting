import os
import sys
import util
import numpy as np
import keras

epochs = 1000
batch_size = 40
num_classes = 20
input_shape = (256, 256, 3)
weight_path = 'modle.h5'

#util.extract_video('data/validate/*.mp4', 5, input_shape)

print('building modle...')
model = util.build_modle(input_shape=input_shape, num_classes=num_classes)

if os.path.exists(weight_path):
    print('load model weight...')
    model.load_weights(weight_path)

if '--train' in sys.argv:
    print('trainning data...')
    util.train(model, 'data/image/*.*', epochs=epochs, input_shape=input_shape, num_classes=num_classes, batch_size=batch_size, weight_path=weight_path)

print('predict data...')
util.predict(model, 'data/validate/*.*', input_shape=input_shape, num_classes=num_classes)
