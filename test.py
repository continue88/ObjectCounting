import os
import glob
import cv2
import util
import numpy as np

def test_random_atals(image_path='data/item/*.png', item_size=(256, 256), atlas_size=(1024, 1024), scale=0.2):
    grid = int(atlas_size[0] / item_size[0])
    image_list = util.load_images(image_path)
    while True:
        row_images = []
        for i in range(grid):
            images = []
            for j in range(grid):
                item_num = np.random.randint(1, 20)
                img = util.build_image(image_list, item_size, scale, item_num)
                images.append(img)
            row_images.append(cv2.hconcat(images))
        image = cv2.vconcat(row_images)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        util.draw_text(image, 'press [esc] to exit, other key to refresh.', color=(255, 0, 0))
        cv2.imshow('image', image)
        key = cv2.waitKey(1000000) & 0xFF
        if key == 27: # 'esc
            break
        util.draw_text(image, 'building image, please wait...', pos=(10, 35), color=(0, 0, 255))
        cv2.imshow('image', image)
        cv2.waitKey(50)

test_random_atals()