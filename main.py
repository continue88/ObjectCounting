import os
import sys
import util
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

epochs = 1000
epoch_save = 10
load_num = 200
batch_size = 40
num_classes = 20
input_shape = (256, 256, 3)
weight_path = 'modle.h5'

#util.extract_video('data/validate/*.mp4', 5, input_shape)

print('building modle...')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

if os.path.exists(weight_path):
    print('load model weight...')
    model.load_weights(weight_path)

if '--train' in sys.argv:
    print('load the trainning data...')
    for epoch in range(epochs):
        data_set = util.load_dataset('data/image/*.*', load_num, input_shape)
        x_train = np.array(data_set[0]).astype(np.float32) / 255.0
        y_train = keras.utils.to_categorical(data_set[1], num_classes)

        train_num = x_train.shape[0]
        indices = np.arange(train_num)

        np.random.shuffle(indices)
        n_batches = int((train_num + batch_size - 1) / batch_size)
        for batch in range(n_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            if end > train_num:
                end = train_num
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            loss = model.train_on_batch(x_batch, y_batch)
            print("[Epoch %d/%d] [Batch %d/%d] [D loss %f, acc: %3f%%]" % (epoch, epochs, batch, n_batches, loss[0], loss[1] * 100))
        if epoch % epoch_save == 0:
            model.save_weights(weight_path)

print('load the test data...')
test_set = util.load_dataset('data/validate/*.*', load_num, input_shape, False)
x_test = np.array(test_set[0]).astype(np.float32) / 255.0
y_test = keras.utils.to_categorical(test_set[1], num_classes)

print('predict data...')
result = model.predict(x_test)
for i in range(len(result)):
    value = np.argmax(result[i])
    print('predict %d, actual %d' % (value, test_set[1][i]))

print('test the trainning data...')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy: %f', score[1])
