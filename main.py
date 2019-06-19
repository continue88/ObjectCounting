import os
import sys
import util
import numpy as np
import keras
from keras.callbacks import TensorBoard

epochs = 1000
epoch_batch = 10
epoch_save = 10
load_num = 600
batch_size = 200
num_classes = 20
input_shape = (256, 256, 3)
model_type = 'default' # ['vgg11', 'vgg16', 'minist', 'default']
weight_path = 'modle-%s.h5' % model_type
log_dir = 'log'

#util.extract_video('data/validate/*.mp4', 5, input_shape)

print('building modle...')
model = util.build_modle(model_type=model_type, input_shape=input_shape, num_classes=num_classes)

if os.path.exists(weight_path):
    print('load model weight...')
    model.load_weights(weight_path)

if '--train' in sys.argv:
    print('trainning data...')
    
    tboard = None
    if '--tb' in sys.argv:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        print('using tensor board at: ' + log_dir)
        tboard = TensorBoard(log_dir, batch_size=epoch_batch,write_grads=True)
        tboard.set_model(model)

    util.train(model, 'data/image/*.*', 
        epochs=epochs, 
        epoch_batch=epoch_batch,
        epoch_save=epoch_save, 
        input_shape=input_shape, 
        num_classes=num_classes, 
        batch_size=batch_size, 
        load_num=load_num, 
        weight_path=weight_path,
        tboard=tboard)

print('predict data...')
util.predict(model, 'data/image/*.*', input_shape=input_shape, num_classes=num_classes)
