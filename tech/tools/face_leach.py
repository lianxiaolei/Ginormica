# -*- coding: utf-8 -*-

import os
import cv2
import dlib
import random


def relight(img, light=1, bias=0):
    """
    改变图片的亮度与对比度
    :param img:
    :param light:
    :param bias:
    :return:
    """
    w = img.shape[1]
    h = img.shape[0]
    # image = []
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


def get_face_from_camera(img_num, output_dir='../../assets/my_faces', size=64):
    """
    调用摄像头获取人脸图像并保存
    :param img_num:
    :param output_dir:
    :param size:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 使用dlib自带的frontal_face_detector作为我们的特征提取器
    detector = dlib.get_frontal_face_detector()
    # 打开摄像头 参数为输入流，可以为摄像头或视频文件
    camera = cv2.VideoCapture(0)

    index = 0
    while True:
        if index < img_num:
            # 从摄像头读取照片
            success, img = camera.read()
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

                face = cv2.resize(face, (size, size))

                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                index += 1
        else:
            break


def get_face_from_lfw(img_num, input_dir, output_dir='../../assets/my_faces', size=64):
    """
    从lfw人脸数据集获取人脸数据
    :param img_num:
    :param input_dir:
    :param output_dir:
    :param size:
    :return:
    """
    # 使用dlib自带的frontal_face_detector作为我们的特征提取器
    detector = dlib.get_frontal_face_detector()

    index = 0
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                print('Being processed picture %s' % index)
                img_path = path + '/' + filename
                # 从文件读取图片
                img = cv2.imread(img_path)
                # 转为灰度图片
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用detector进行人脸检测 dets为返回的结果
                dets = detector(gray_img, 1)

                # 使用enumerate 函数遍历序列中的元素以及它们的下标
                # 下标i即为人脸序号
                # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
                # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    # img[y:y+h,x:x+w]
                    face = img[x1:y1, x2:y2]
                    # 调整图片的尺寸
                    face = cv2.resize(face, (size, size))
                    cv2.imshow('image', face)
                    # 保存图片
                    cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                    index += 1
        if index >= img_num:
            break
