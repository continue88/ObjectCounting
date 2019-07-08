import os
import cv2
import glob
import numpy as np
import keras
import threading
import time
import platform
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Reshape
from keras.layers import Flatten, UpSampling2D
from keras import backend as K
from pixel_shuffler import PixelShuffler

def ObjectCountingLoss(y_true, y_pred):
    scale = K.clip((1 - y_true) * 255, 0, 1) + 1
    return K.mean(K.square(y_pred - y_true) * scale, axis=-1)

def build_h128(input_shape, num_classes, features = 32, alpha = 0.1, loop = 4):
    input = Input(input_shape)
    x = input # [256, 256, 3]
    x = LeakyReLU(alpha)(Conv2D(features *  1, kernel_size=5, strides=2, padding='same')(x)) # [128, 128,   64]
    x = LeakyReLU(alpha)(Conv2D(features *  2, kernel_size=5, strides=2, padding='same')(x)) # [ 64,  64,  128]
    x = LeakyReLU(alpha)(Conv2D(features *  4, kernel_size=5, strides=2, padding='same')(x)) # [ 32,  32,  256]
    x = LeakyReLU(alpha)(Conv2D(features *  8, kernel_size=5, strides=2, padding='same')(x)) # [ 16,  16,  512]
    x = LeakyReLU(alpha)(Conv2D(features * 16, kernel_size=5, strides=2, padding='same')(x)) # [  8,   8, 1024]
    x = Dense(512)(Flatten()(x))
    x = Dense(8 * 8 * 1024)(x)
    x = Reshape((8, 8, 1024))(x) # [8, 8, 1024]
    x = PixelShuffler()(LeakyReLU(0.1)(Conv2D(512 * 4, 3, strides=1, padding='same')(x))) # [16, 16, 512]
    x = PixelShuffler()(LeakyReLU(0.1)(Conv2D(256 * 4, 3, strides=1, padding='same')(x))) # [32, 32, 256]
    x = PixelShuffler()(LeakyReLU(0.1)(Conv2D(128 * 4, 3, strides=1, padding='same')(x))) # [64, 64, 128]
    x = PixelShuffler()(LeakyReLU(0.1)(Conv2D(64 * 4, 3, strides=1, padding='same')(x))) # [128, 128, 64]
    x = PixelShuffler()(LeakyReLU(0.1)(Conv2D(32 * 4, 3, strides=1, padding='same')(x))) # [256, 256, 32]
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x) # (256, 256, 3)
    model = Model(input, x)
    model.compile(loss=ObjectCountingLoss,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    return model
    
def build_modle(input_shape=(256, 256, 3), num_classes=20, model_type='vgg16'):
    return build_h128(input_shape, num_classes)

class ImageGenerator(threading.Thread):
    ''' The image builder thread. load single item images, gennerate images.
    '''
    def __init__(self, data_path, size, scale, num_classes, load_num, val_num=6):
        threading.Thread.__init__(self)
        self.data_path = data_path
        self.size = size
        self.scale = scale
        self.num_classes = num_classes
        self.load_num = load_num
        self.val_num = val_num
        self.stop = False
        self.data_set = None
        self.validate = None
        self.thread_lock = threading.Lock()
    
    def fetch_images(self):
        data_set = None
        while not data_set:
            if self.data_set:
                self.thread_lock.acquire()
                data_set = self.data_set
                self.data_set = None
                self.thread_lock.release()
            else:
                time.sleep(0.1)
        return data_set
    
    def join(self):
        self.stop = True
        threading.Thread.join(self)

    def run(self):
        self.image_list = load_images(self.data_path)
        self.validate = build_image_set(self.image_list, self.size, self.scale, self.val_num)
        while not self.stop:
            if not self.data_set:
                scale = self.scale + np.random.random() * 0.05
                data_set = random_dataset(self.image_list, self.size, scale, self.num_classes, self.load_num)
                self.thread_lock.acquire()
                self.data_set = data_set
                self.thread_lock.release()
            else:
                time.sleep(0.1)

def train(model, data_path, epochs=1000, load_num=200, input_shape=(256, 256, 3), num_classes=20, batch_size=40, validate=10, epoch_save=100, weight_path='modle.h5', tboard=None):
    # build our image generator.
    size = (input_shape[0], input_shape[1])
    image_generator = ImageGenerator(data_path, size, 0.2, num_classes, load_num)
    image_generator.start()

    # start training...
    for epoch in range(epochs):
        if (epoch + 1) % epoch_save == 0:
            model.save_weights(weight_path)

        data_set = image_generator.fetch_images()
        x_train = data_set[0]
        y_train = data_set[1]
        loss = None

        if platform.system() == 'Windows' and epoch % validate == 0:
            x_test = image_generator.validate[0]
            y_test = image_generator.validate[1]
            y_predict = model.predict(np.array([x_test]))
            img_test = cv2.hconcat([x_test, y_test, y_predict[0]])
            img_test = np.clip(img_test * 255, 0, 255).astype(np.uint8)
            cv2.imshow('image', img_test)
            cv2.waitKey(10)
        
        train_num = len(x_train)#.shape[0]
        n_batches = int((train_num + batch_size - 1) / batch_size)
        for batch in range(n_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            if end > train_num:
                end = train_num
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            loss = model.train_on_batch(x_batch, y_batch)
            if batch == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss %f, acc: %3f%%]" % (epoch, epochs, batch, n_batches, loss[0], loss[1] * 100))

        if tboard:
            tboard.on_epoch_end(epoch, {'loss': loss[0], 'acc': loss[1]})
    # finished.
    if tboard:
        tboard.on_train_end('done')
    model.save_weights(weight_path)
    image_generator.join()

def predict(model, validate_path, input_shape=(256, 256, 3), num_classes=20, load_num=200):
    test_set = load_dataset(validate_path, load_num, input_shape)
    x_test = np.array(test_set[0]).astype(np.float32) / 255.0
    y_test = keras.utils.to_categorical(test_set[1], num_classes)

    result = model.predict(x_test)
    for i in range(len(result)):
        value = np.argmax(result[i])
        print('predict %d, actual %d' % (value, test_set[1][i]))

    print('test the trainning data...')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy: %f%%' % (score[1] * 100))

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

def rotate_images(image_list, angles=None):
    rot_images = []
    for img in image_list:
        if angles != None:
            angle = angles[np.random.randint(0, len(angles))]
        else:
            angle = np.random.randint(0, 360)
        img = rotate_image(img, angle)
        rot_images.append(img)
    return np.array(rot_images)

def load_dataset(path, load_num, input_shape):
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
        img_list.append(img)
        label_list.append(label)
    return (img_list, label_list)

def crop_video(input, output, resize):
    input = cv2.VideoCapture(input)
    output = cv2.VideoWriter(output, int(input.get(cv2.CAP_PROP_FOURCC)), input.get(cv2.CAP_PROP_FPS), resize)
    success, img = input.read()
    while success:
        # 裁剪成正方形
        height, width, _ = img.shape
        size = min(height, width)
        x = int((width - size) / 2)
        y = int((height - size) / 2)
        img = img[y:y+size, x:x+size]
        # 转换到网络输入
        img = cv2.resize(img, resize, 0, 0, cv2.INTER_CUBIC)
        output.write(img)
        success, img = input.read()
    input.release()
    output.release()

def random_image(base_size, image, scale, large_scale = 2):
    large_size = (base_size[0] * large_scale, base_size[1] * large_scale)
    scale = scale * large_scale
    imge_size = image.shape[:2]
    edge = (imge_size[0] * scale * 0.5, imge_size[1] * scale * 0.5)
    pos = (np.random.randint(edge[0], large_size[0] - edge[0]), np.random.randint(edge[1], large_size[1] - edge[1]))
    center = (imge_size[0] * 0.5, imge_size[1] * 0.5)
    angle = np.random.rand() * 360
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[:, 2] += (pos[0] - center[0], pos[1] - center[1])
    new_img = cv2.warpAffine(image, rot_mat, large_size, cv2.INTER_LINEAR, 0, cv2.BORDER_REPLICATE)
    new_img = cv2.resize(new_img, (base_size[1], base_size[0]))
    return (new_img, float(pos[0]) / large_size[0], float(pos[1]) / large_size[1], angle / 360.0)

def build_image(image_list, size, scale, item_num):
    ''' 随机组合n张小图片成一张大图片
    '''
    base_img = np.zeros((size[0], size[1], 4))
    for _ in range(item_num):
        image_idx = np.random.randint(0, len(image_list))
        img_data = random_image(base_img.shape[:2], image_list[image_idx], scale)
        random_img = img_data[0]
        alpha = random_img[:,:,3]
        alpha = np.repeat(np.expand_dims(alpha, -1), (4,), -1)
        base_img = base_img * (1 - alpha) + alpha * random_img
    return base_img[:,:,0:3]

def build_image_set(image_list, size, scale, item_num):
    ''' 随机组合n张小图片成一张大图片
    '   返回原图，位置/旋转图（同样大小）
    '''
    base_img = np.zeros((size[0], size[1], 4), np.float32)
    pos_img = np.zeros((size[0], size[1], 4), np.float32)
    for _ in range(item_num):
        image_idx = np.random.randint(0, len(image_list))
        img_data = random_image(base_img.shape[:2], image_list[image_idx], scale)
        random_img = img_data[0]
        alpha = random_img[:,:,3]
        alpha = np.repeat(np.expand_dims(alpha, -1), (4,), -1)
        base_img = base_img * (1 - alpha) + alpha * random_img
        pos_img = pos_img * (1 - alpha) + alpha * np.float32([img_data[1], img_data[2], img_data[3], 1])
    return (base_img[:,:,0:3], pos_img[:,:,0:3])

def random_dataset(image_list, size, scale, num_classes, total_num):
    ''' 随机生成数据集合
    '''
    data_list = []
    label_list = []
    for _ in range(total_num):
        item_num = np.random.randint(1, num_classes)
        img = build_image_set(image_list, size, scale, item_num)
        data_list.append(img[0])
        label_list.append(img[1])
    return (np.array(data_list), np.array(label_list))

def load_images(path, uv=False):
    file_list = glob.glob(path)
    image_list = []
    for file in file_list:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if uv:
            height, width, _ = img.shape
            for i in range(width):
                for j in range(height):
                    img[i, j, :3] = (float(i) / width, float(j) / width, 0)
        else:
            img = img.astype(np.float32) / 255.0
        image_list.append(img)
    return image_list

def draw_text(image, text, pos=(10, 20), font_scale=0.5, color=(255, 255, 255), shadow=True):
    if shadow:
        cv2.putText(image, text, (pos[0] + 0, pos[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0))
        cv2.putText(image, text, (pos[0] + 1, pos[1] + 0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0))
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)