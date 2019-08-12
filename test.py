import os
import glob
import cv2
import util
import numpy as np

image_path = 'data/item/*.png'

def test_random_atals(image_path, item_size=(256, 256), atlas_size=(1024, 1024), scale=0.2):
    grid = int(atlas_size[0] / item_size[0])
    image_list = util.load_images(image_path)
    while True:
        row_images = []
        for i in range(grid):
            images = []
            for j in range(grid):
                item_num = np.random.randint(1, 20)
                tmp_scale = scale + np.random.random() * 0.05
                img = util.build_image(image_list, item_size, tmp_scale, item_num)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                util.draw_text(img, '%d' % item_num, color=(255, 255, 255))
                images.append(img)
            row_images.append(cv2.hconcat(images))
        image = cv2.vconcat(row_images)
        util.draw_text(image, 'press [esc] to exit, other key to refresh.', color=(255, 0, 0))
        cv2.imshow('image', image)
        key = cv2.waitKey(1000000) & 0xFF
        if key == 27: # 'esc
            break
        util.draw_text(image, 'building image, please wait...', pos=(10, 35), color=(0, 0, 255))
        cv2.imshow('image', image)
        cv2.waitKey(50)

def test_rotate_image(image_path, item_size=(256, 256), scale=0.2, item_num=6):
    image_list = util.load_images(image_path)
    image = util.build_image(image_list, item_size, scale, item_num)
    pixels = image.reshape(-1, image.shape[2])
    unique_elements, counts_elements = np.unique(pixels, axis=0, return_counts=True)
    counts_elements = np.sort(counts_elements)[::-1]
    angles = [0, 90, 180, 270]
    while True:
        angle = angles[np.random.randint(0, len(angles))]
        rot_image = np.clip(image * 255, 0, 255).astype(np.uint8)
        rot_image = util.rotate_image(rot_image, angle)
        util.draw_text(rot_image, 'angle: %d' % angle, color=(255, 0, 0))
        cv2.imshow('image', rot_image)
        key = cv2.waitKey(1000000) & 0xFF
        if key == 27: # 'esc
            break

test_random_atals(image_path)
#test_rotate_image(image_path)