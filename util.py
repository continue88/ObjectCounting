import os
import cv2
import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_modle(input_shape=(256, 256, 3), num_classes=20):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
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
    return model

def train(model, data_path, epochs=1000, load_num=200, input_shape=(256, 256, 3), num_classes=20, batch_size=40, epoch_save=100, weight_path='modle.h5'):
    for epoch in range(epochs):
        if (epoch + 1) % epoch_save == 0:
            model.save_weights(weight_path)

        data_set = load_dataset(data_path, load_num, input_shape)
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
    model.save_weights(weight_path)

def predict(model, validate_path, input_shape=(256, 256, 3), num_classes=20, load_num=200):
    test_set = load_dataset(validate_path, load_num, input_shape, False)
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
    for file in file_list:
        file_name = os.path.basename(file)
        end = file_name.rfind(' ')
        if end < 0:
            end = file_name.rfind('.')
        label = int(file_name[:end])
        img = cv2.imread(file)
        img = cv2.resize(img, (width, height))
        if train:
            angle = np.random.randint(0, 360)
            img = rotate_image(img, angle)
        img_list.append(img)
        label_list.append(label)
    return (img_list, label_list)
