import threading
import time
import numpy as np
import util

class DataBuilder(threading.Thread):
    ''' The image data builder thread. load single item images, gennerate images.
    '''
    def __init__(self, data_path, size=(256, 256), num_classes=20, load_num=200, scale=0.2):
        threading.Thread.__init__(self)
        self.data_path = data_path
        self.size = size
        self.scale = scale
        self.num_classes = num_classes
        self.load_num = load_num
        self.stop = False
        self.data_set = None
        self.pending_data = None
        self.pending_index = 0
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

    def load_data(self, batch_size):
        if not self.pending_data:
            self.pending_data = self.fetch_images()
        end = min(self.pending_index + batch_size, self.load_num)
        data_set = (self.pending_data[0][self.pending_index:end], 
            self.pending_data[1][self.pending_index:end])
        self.pending_index += batch_size
        if end == self.load_num:
            self.pending_data = None
        return data_set

    def random_dataset(self, scale):
        data_list = []
        label_list = []
        for _ in range(self.load_num):
            item_num = np.random.randint(1, self.num_classes)
            img_set = util.build_image_set(self.image_list, self.size, scale, item_num)
            data_list.append(img_set[0])
            label_list.append(img_set[1])
        return (np.array(data_list), np.array(label_list))

    
    def join(self):
        self.stop = True
        threading.Thread.join(self)

    def run(self):
        self.image_list = util.load_images(self.data_path)
        
        while not self.stop:
            if not self.data_set:
                scale = self.scale + np.random.random() * 0.05
                data_set = self.random_dataset(scale)
                self.thread_lock.acquire()
                self.data_set = data_set
                self.thread_lock.release()
            else:
                time.sleep(0.1)