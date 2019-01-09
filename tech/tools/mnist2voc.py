# coding: utf-8

import cv2
import numpy as np
import os
from xml.dom import minidom
import random as rd
import time
import matplotlib.pyplot as plt
import sys


def find_text_region(img):
    """
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray[gray > 120] = 255
    gray[gray <= 120] = 0

    # region = find_text_region(gray)

    region = []

    # 1. 查找轮廓
    image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 画图像的边轮廓
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]

        rect = cv2.boundingRect(cnt)

        region.append(rect)

    return region[0]


def mnist2voc(in_path, out_path, size=(224, 224), test_ratio=0.2):
    """

    :param in_path:
    :param out_path:
    :param size:
    :return:
    """
    depth = 3
    # make dirs
    anno_dir = os.path.join(out_path, 'Annotations')
    if 'Annotations' not in os.listdir(out_path):
        os.mkdir(anno_dir)

    set_dir = os.path.join(out_path, 'ImageSets')
    if 'ImageSets' not in os.listdir(out_path):
        os.mkdir(set_dir)
    if 'Main' not in os.listdir(set_dir):
        os.mkdir(os.path.join(set_dir, 'Main'))
    set_dir = os.path.join(set_dir, 'Main')

    jpg_dir = os.path.join(out_path, 'JPEGImages')
    if 'JPEGImages' not in os.listdir(out_path):
        os.mkdir(jpg_dir)

    tmp_time = time.time()
    for epoch in range(20000):
        img_name = '0' * (6 - len(str(epoch))) + str(epoch)
        jpg_name = img_name + '.jpg'
        xml_name = img_name + '.xml'
        xml = minidom.getDOMImplementation()
        doc = xml.createDocument(None, None, None)

        root = doc.createElement('annotation')

        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode(jpg_name))

        sz = doc.createElement('size')
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(size[0])))
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(size[1])))
        dp = doc.createElement('depth')
        dp.appendChild(doc.createTextNode(str(depth)))
        sz.appendChild(width)
        sz.appendChild(height)
        sz.appendChild(dp)
        root.appendChild(filename)
        root.appendChild(sz)

        out_img = np.zeros((size[1], size[0], depth), dtype=np.uint8)

        num = rd.randint(10, 30)
        for i in range(num):
            rnd_numdir = rd.randint(0, 9)
            fnames = os.listdir(os.path.join(in_path, str(rnd_numdir)))
            rnd_fname = rd.choice(fnames)
            long_fname = os.path.join(in_path, str(rnd_numdir), rnd_fname)

            img = cv2.imread(long_fname)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            zoom_ratio = rd.randrange(8, 12, 1) / 10
            rotate_ratio = rd.randrange(-45, 45, 10)
            m = cv2.getRotationMatrix2D(center, rotate_ratio, zoom_ratio)
            conversion = cv2.warpAffine(img, m, (w, h))
            conversion = cv2.resize(conversion, dsize=(w // 2, h // 2))
            h, w = conversion.shape[:2]

            rnd_x, rnd_y = rd.randrange(0, size[1] - w, w), rd.randrange(0, size[0] - h, h)
            # print('rnd_y %s,h %s,rnd_x %s,w %s,conversion %s' % (rnd_y, h, rnd_x, w, conversion.shape[:2]))
            out_img[rnd_y: rnd_y + h, rnd_x: rnd_x + w, :] = conversion

            obj = doc.createElement('object')
            name = doc.createElement('name')
            name.appendChild(doc.createTextNode(str(rnd_numdir)))
            bndbox = doc.createElement('bndbox')
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(str(rnd_x)))
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(str(rnd_y)))
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str(rnd_x + w)))
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str(rnd_y + h)))
            bndbox.appendChild(xmin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymin)
            bndbox.appendChild(ymax)
            diff = doc.createElement('difficult')
            diff.appendChild(doc.createTextNode('0'))
            obj.appendChild(name)
            obj.appendChild(bndbox)
            obj.appendChild(diff)
            root.appendChild(obj)

        # num_noi = 1000
        # for k in range(num_noi):
        #     # get the random point
        #     xi = int(np.random.uniform(0, out_img.shape[1]))
        #     xj = int(np.random.uniform(0, out_img.shape[0]))
        #     # add noise
        #     no0, no1, no2 = rd.randint(20, 255), rd.randint(20, 255), rd.randint(20, 255)
        #     out_img[xj, xi, 0] = no0
        #     out_img[xj, xi, 1] = no1
        #     out_img[xj, xi, 2] = no2

        cv2.imwrite(os.path.join(jpg_dir, jpg_name), out_img)
        f = open(os.path.join(anno_dir, xml_name), 'w')
        f.write(root.toprettyxml(encoding='utf8').decode('utf8'))
        f.close()
        rnd = rd.randint(1, 10)
        if rnd > (10 - 10 * test_ratio):
            file2write = 'test.txt'
        else:
            file2write = 'trainval.txt'
        f = open(os.path.join(set_dir, file2write), 'a')
        f.write(img_name + '\n')
        f.close()

        if (not epoch == 0) and (epoch % 200 == 0):
            print('operated %s files done. %s seconds cost.' % (epoch, time.time() - tmp_time))


if __name__ == '__main__':
    mnist2voc('F:\\num_ocr', 'F:\\mnist_voc\\VOC2007')
