# -*- coding:utf-8 -*-
from absl import flags
from random import shuffle, random
from model_V3 import *

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# FPN 아이디어까지는 괜찮았지만, loss를 또 어떻게 꾸려야할지가 문제이다.
# Roipooling in mask r-cnn을 GAN에 적용해볼까? 생성함에 있어도 픽셀의 위치는 중요하지 아니한가?
flags.DEFINE_string("A_tr_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/male_16_39_train.txt", "")

flags.DEFINE_string("A_tr_img_path", "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/", "")

flags.DEFINE_string("B_tr_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/female_40_63_train.txt", "")

flags.DEFINE_string("B_tr_img_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/female_40_63/", "")

flags.DEFINE_string("A_nose_txt", "C:/Users/Yuhwan/Downloads/AFAD_nose.txt", "")

flags.DEFINE_string("A_eyes_txt", "C:/Users/Yuhwan/Downloads/AFAD_eyes.txt", "")

flags.DEFINE_string("A_mouth_txt", "C:/Users/Yuhwan/Downloads/AFAD_mouth.txt", "")

flags.DEFINE_string("B_nose_txt", "C:/Users/Yuhwan/Downloads/Morph_nose.txt", "")

flags.DEFINE_string("B_eyes_txt", "C:/Users/Yuhwan/Downloads/Morph_eyes.txt", "")

flags.DEFINE_string("B_mouth_txt", "C:/Users/Yuhwan/Downloads/Morph_mouth.txt", "")

flags.DEFINE_integer("img_size", 256, "")

flags.DEFINE_integer("img_ch", 3, "")

flags.DEFINE_integer("batch_size", 2, "")

flags.DEFINE_integer("epochs", 200, "")

flags.DEFINE_float("lr", 0.0002, "")

flags.DEFINE_bool("train", True, "")

flags.DEFINE_bool("pre_checkpoint", False, "")

flags.DEFINE_string("pre_checkpoint_path", "", "")

flags.DEFINE_string("sample_images_path", "", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

g_optim = tf.keras.optimizers.Adam(FLAGS.lr)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr)



def nose_coordinate_box(a1, a2, a3, a4):

    xmin, ymin = a2[0], a1[1]
    xmax, ymax = a4[0], a3[1]

    return [ymin, xmin, ymax, xmax]

def eyes_coordinate_box(a1, a2, a3, a4,
                        b1, b2, b3, b4):

    r_xmin, r_ymin = a1[0], a2[1]
    r_xmax, r_ymax = a3[0], a4[1]

    l_xmin, l_ymin = b1[0], b2[1]
    l_xmax, l_ymax = b3[0], b4[1]

    r_box = [r_ymin, r_xmin, r_ymax, r_xmax]
    l_box = [l_ymin, l_xmin, l_ymax, l_xmax]

    return r_box, l_box

def mouth_coordinate_box(a1, a2, a3, a4):

    xmin, ymin = a1[0], a2[1]
    xmax, ymax = a3[0], a4[1]

    return [ymin, xmin, ymax, xmax]

def fasial_lanmark(A_nose,
                   A_eyes,
                   A_mouth,
                   B_nose,
                   B_eyes,
                   B_mouth,
                   A_resize_img,
                   B_resize_img):
    # nose - 9, eyes- 12, mouth - 20
    A_nose = np.loadtxt(A_nose, dtype=np.int32, skiprows=0, usecols=[0, 1])
    A_eyes = np.loadtxt(A_eyes, dtype=np.int32, skiprows=0, usecols=[0, 1])
    A_mouth = np.loadtxt(A_mouth, dtype=np.int32, skiprows=0, usecols=[0, 1])

    B_nose = np.loadtxt(B_nose, dtype=np.int32, skiprows=0, usecols=[0, 1])
    B_eyes = np.loadtxt(B_eyes, dtype=np.int32, skiprows=0, usecols=[0, 1])
    B_mouth = np.loadtxt(B_mouth, dtype=np.int32, skiprows=0, usecols=[0, 1])

    A_nose_128, A_eyes_128, A_mouth_128 = np.ceil(A_nose * (128 / 256)), \
                                          np.ceil(A_eyes * (128 / 256)), \
                                          np.ceil(A_mouth * (128 / 256))
    A_nose_64, A_eyes_64, A_mouth_64 = np.ceil(A_nose * (64 / 256)), \
                                       np.ceil(A_eyes * (64 / 256)), \
                                       np.ceil(A_mouth * (64 / 256))
    A_nose_32, A_eyes_32, A_mouth_32 = np.ceil(A_nose * (32 / 256)), \
                                       np.ceil(A_eyes * (32 / 256)), \
                                       np.ceil(A_mouth * (32 / 256))
    A_nose_16, A_eyes_16, A_mouth_16 = np.ceil(A_nose * (16 / 256)), \
                                       np.ceil(A_eyes * (16 / 256)), \
                                       np.ceil(A_mouth * (16 / 256))

    B_nose_128, B_eyes_128, B_mouth_128 = np.ceil(B_nose * (128 / 256)), \
                                          np.ceil(B_eyes * (128 / 256)), \
                                          np.ceil(B_mouth * (128 / 256))
    B_nose_64, B_eyes_64, B_mouth_64 = np.ceil(B_nose * (64 / 256)), \
                                       np.ceil(B_eyes * (64 / 256)), \
                                       np.ceil(B_mouth * (64 / 256))
    B_nose_32, B_eyes_32, B_mouth_32 = np.ceil(B_nose * (32 / 256)), \
                                       np.ceil(B_eyes * (32 / 256)), \
                                       np.ceil(B_mouth * (32 / 256))
    B_nose_16, B_eyes_16, B_mouth_16 = np.ceil(B_nose * (16 / 256)), \
                                       np.ceil(B_eyes * (16 / 256)), \
                                       np.ceil(B_mouth * (16 / 256))

    # 이 부분에 박스 좌표에 관한코드를 추가해서 넣으면 된다.
    A_nose_128_box = nose_coordinate_box(A_nose_128[0], A_nose_128[4], A_nose_128[6], A_nose_128[8])
    A_nose_64_box = nose_coordinate_box(A_nose_64[0], A_nose_64[4], A_nose_64[6], A_nose_64[8])
    A_nose_32_box = nose_coordinate_box(A_nose_32[0], A_nose_32[4], A_nose_32[6], A_nose_32[8])
    A_nose_16_box = nose_coordinate_box(A_nose_16[0], A_nose_16[4], A_nose_16[6], A_nose_16[8])

    A_eyes_128_box = eyes_coordinate_box(A_eyes_128[0], A_eyes_128[1], A_eyes_128[3], A_eyes_128[4],
                                         A_eyes_128[6], A_eyes_128[7], A_eyes_128[9], A_eyes_128[10])
    A_eyes_64_box = eyes_coordinate_box(A_eyes_64[0], A_eyes_64[1], A_eyes_64[3], A_eyes_64[4],
                                        A_eyes_64[6], A_eyes_64[7], A_eyes_64[9], A_eyes_64[10])
    A_eyes_32_box = eyes_coordinate_box(A_eyes_32[0], A_eyes_32[1], A_eyes_32[3], A_eyes_32[4],
                                        A_eyes_32[6], A_eyes_32[7], A_eyes_32[9], A_eyes_32[10])
    A_eyes_16_box = eyes_coordinate_box(A_eyes_16[0], A_eyes_16[1], A_eyes_16[3], A_eyes_16[4],
                                        A_eyes_16[6], A_eyes_16[7], A_eyes_16[9], A_eyes_16[10])

    A_mouth_128_box = mouth_coordinate_box(A_mouth_128[0], A_mouth_128[2], A_mouth_128[6], A_mouth_128[8])
    A_mouth_64_box = mouth_coordinate_box(A_mouth_64[0], A_mouth_64[2], A_mouth_64[6], A_mouth_64[8])
    A_mouth_32_box = mouth_coordinate_box(A_mouth_32[0], A_mouth_32[2], A_mouth_32[6], A_mouth_32[8])
    A_mouth_16_box = mouth_coordinate_box(A_mouth_16[0], A_mouth_16[2], A_mouth_16[6], A_mouth_16[8])

    B_nose_128_box = nose_coordinate_box(B_nose_128[0], B_nose_128[4], B_nose_128[6], B_nose_128[8])
    B_nose_64_box = nose_coordinate_box(B_nose_64[0], B_nose_64[4], B_nose_64[6], B_nose_64[8])
    B_nose_32_box = nose_coordinate_box(B_nose_32[0], B_nose_32[4], B_nose_32[6], B_nose_32[8])
    B_nose_16_box = nose_coordinate_box(B_nose_16[0], B_nose_16[4], B_nose_16[6], B_nose_16[8])

    B_eyes_128_box = eyes_coordinate_box(B_eyes_128[0], B_eyes_128[1], B_eyes_128[3], B_eyes_128[4],
                                         B_eyes_128[6], B_eyes_128[7], B_eyes_128[9], B_eyes_128[10])
    B_eyes_64_box = eyes_coordinate_box(B_eyes_64[0], B_eyes_64[1], B_eyes_64[3], B_eyes_64[4],
                                        B_eyes_64[6], B_eyes_64[7], B_eyes_64[9], B_eyes_64[10])
    B_eyes_32_box = eyes_coordinate_box(B_eyes_32[0], B_eyes_32[1], B_eyes_32[3], B_eyes_32[4],
                                        B_eyes_32[6], B_eyes_32[7], B_eyes_32[9], B_eyes_32[10])
    B_eyes_16_box = eyes_coordinate_box(B_eyes_16[0], B_eyes_16[1], B_eyes_16[3], B_eyes_16[4],
                                        B_eyes_16[6], B_eyes_16[7], B_eyes_16[9], B_eyes_16[10])

    B_mouth_128_box = mouth_coordinate_box(B_mouth_128[0], B_mouth_128[2], A_mouth_128[6], A_mouth_128[8])
    B_mouth_64_box = mouth_coordinate_box(B_mouth_64[0], B_mouth_64[2], B_mouth_64[6], B_mouth_64[8])
    B_mouth_32_box = mouth_coordinate_box(B_mouth_32[0], B_mouth_32[2], B_mouth_32[6], B_mouth_32[8])
    B_mouth_16_box = mouth_coordinate_box(B_mouth_16[0], B_mouth_16[2], B_mouth_16[6], B_mouth_16[8])
    
    A_nose_value = np.zeros([FLAGS.batch_size, 9], dtype=np.float32)
    A_eyes_value = np.zeros([FLAGS.batch_size, 12], dtype=np.float32)
    A_mouth_value = np.zeros([FLAGS.batch_size, 20], dtype=np.float32)

    B_nose_value = np.zeros([FLAGS.batch_size, 9], dtype=np.float32)
    B_eyes_value = np.zeros([FLAGS.batch_size, 12], dtype=np.float32)
    B_mouth_value = np.zeros([FLAGS.batch_size, 20], dtype=np.float32)

    A_gray = A_resize_img[0]
    B_gray = B_resize_img[0]
    for j in range(FLAGS.batch_size):
        for i in range(9):
            A_n = A_nose[i]
            B_n = B_nose[i]
            A_nose_value[j,i] = A_gray[j, A_n[1], A_n[0], 0]
            B_nose_value[j,i] = B_gray[j, B_n[1], B_n[0], 0]
        for i in range(12):
            A_e = A_eyes[i]
            B_e = B_eyes[i]
            A_eyes_value[j,i] = A_gray[j, A_e[1], A_e[0], 0]
            B_eyes_value[j,i] = B_gray[j, B_e[1], B_e[0], 0]
        for i in range(20):
            A_m = A_mouth[i]
            B_m = B_mouth[i]
            A_mouth_value[j,i] = A_gray[j, A_m[1], A_m[0], 0]
            B_mouth_value[j,i] = B_gray[j, B_m[1], B_m[0], 0]

    A_value = (A_nose_value, A_eyes_value, A_mouth_value)
    B_value = (B_nose_value, B_eyes_value, B_mouth_value)

    A_nose_coord = (A_nose_128, A_nose_64, A_nose_32, A_nose_16)
    A_eyes_coord = (A_eyes_128, A_eyes_64, A_eyes_32, A_eyes_16)
    A_mouth_coord = (A_mouth_128, A_mouth_64, A_mouth_32, A_mouth_16)

    B_nose_coord = (B_nose_128, B_nose_64, B_nose_32, B_nose_16)
    B_eyes_coord = (B_eyes_128, B_eyes_64, B_eyes_32, B_eyes_16)
    B_mouth_coord = (B_mouth_128, B_mouth_64, B_mouth_32, B_mouth_16)

    A_nose_box = (A_nose_128_box, A_nose_64_box, A_nose_32_box, A_nose_16_box)
    A_eyes_box = (A_eyes_128_box, A_eyes_64_box, A_eyes_32_box, A_eyes_16_box)
    A_mouth_box = (A_mouth_128_box, A_mouth_64_box, A_mouth_32_box, A_mouth_16_box)

    B_nose_box = (B_nose_128_box, B_nose_64_box, B_nose_32_box, B_nose_16_box)
    B_eyes_box = (B_eyes_128_box, B_eyes_64_box, B_eyes_32_box, B_eyes_16_box)
    B_mouth_box = (B_mouth_128_box, B_mouth_64_box, B_mouth_32_box, B_mouth_16_box)

    #return (A_value, A_nose_128, A_nose_64, A_nose_32, A_nose_16), (B_value, B_nose_128, B_nose_64, B_nose_32, B_nose_16)
    return A_value, B_value, A_nose_coord, A_eyes_coord, A_mouth_coord, B_nose_coord, B_eyes_coord, B_mouth_coord,\
           A_nose_box, A_eyes_box, A_mouth_box, B_nose_box, B_eyes_box, B_mouth_box

def crop_image_part(A_nose_box, A_eyes_box, A_mouth_box, B_nose_box, B_eyes_box, B_mouth_box,
                    A_128, A_64, A_32, A_16, B_128, B_64, B_32, B_16):

    real_A_128_nose = tf.image.crop_to_bounding_box(A_128, int(A_nose_box[0][0]), int(A_nose_box[0][1]), int(A_nose_box[0][2] - A_nose_box[0][0]), int(A_nose_box[0][3] - A_nose_box[0][1]))
    real_A_64_nose = tf.image.crop_to_bounding_box(A_64, int(A_nose_box[1][0]), int(A_nose_box[1][1]), int(A_nose_box[1][2] - A_nose_box[1][0]), int(A_nose_box[1][3] - A_nose_box[1][1]))
    real_A_32_nose = tf.image.crop_to_bounding_box(A_32, int(A_nose_box[2][0]), int(A_nose_box[2][1]), int(A_nose_box[2][2] - A_nose_box[2][0]), int(A_nose_box[2][3] - A_nose_box[2][1]))
    real_A_16_nose = tf.image.crop_to_bounding_box(A_16, int(A_nose_box[3][0]), int(A_nose_box[3][1]), int(A_nose_box[3][2] - A_nose_box[3][0]), int(A_nose_box[3][3] - A_nose_box[3][1]))

    real_A_128_Reyes = tf.image.crop_to_bounding_box(A_128, int(A_eyes_box[0][0][0]), int(A_eyes_box[0][0][1]), int(A_eyes_box[0][0][2] - A_eyes_box[0][0][0]), int(A_eyes_box[0][0][3] - A_eyes_box[0][0][1]))
    real_A_64_Reyes = tf.image.crop_to_bounding_box(A_64, int(A_eyes_box[1][0][0]), int(A_eyes_box[1][0][1]), int(A_eyes_box[1][0][2] - A_eyes_box[1][0][0]), int(A_eyes_box[1][0][3] - A_eyes_box[1][0][1]))
    real_A_32_Reyes = tf.image.crop_to_bounding_box(A_32, int(A_eyes_box[2][0][0]), int(A_eyes_box[2][0][1]), int(A_eyes_box[2][0][2] - A_eyes_box[2][0][0]), int(A_eyes_box[2][0][3] - A_eyes_box[2][0][1]))
    real_A_16_Reyes = tf.image.crop_to_bounding_box(A_16, int(A_eyes_box[3][0][0]), int(A_eyes_box[3][0][1]), int(A_eyes_box[3][0][2] - A_eyes_box[3][0][0]), int(A_eyes_box[3][0][3] - A_eyes_box[3][0][1]))

    real_A_128_Leyes = tf.image.crop_to_bounding_box(A_128, int(A_eyes_box[0][1][0]), int(A_eyes_box[0][1][1]), int(A_eyes_box[0][1][2] - A_eyes_box[0][1][0]), int(A_eyes_box[0][1][3] - A_eyes_box[0][1][1]))
    real_A_64_Leyes = tf.image.crop_to_bounding_box(A_64, int(A_eyes_box[1][1][0]), int(A_eyes_box[1][1][1]), int(A_eyes_box[1][1][2] - A_eyes_box[1][1][0]), int(A_eyes_box[1][1][3] - A_eyes_box[1][1][1]))
    real_A_32_Leyes = tf.image.crop_to_bounding_box(A_32, int(A_eyes_box[2][1][0]), int(A_eyes_box[2][1][1]), int(A_eyes_box[2][1][2] - A_eyes_box[2][1][0]), int(A_eyes_box[2][1][3] - A_eyes_box[2][1][1]))
    real_A_16_Leyes = tf.image.crop_to_bounding_box(A_16, int(A_eyes_box[3][1][0]), int(A_eyes_box[3][1][1]), int(A_eyes_box[3][1][2] - A_eyes_box[3][1][0]), int(A_eyes_box[3][1][3] - A_eyes_box[3][1][1]))

    real_A_128_mouth = tf.image.crop_to_bounding_box(A_128, int(A_mouth_box[0][0]), int(A_mouth_box[0][1]), int(A_mouth_box[0][2] - A_mouth_box[0][0]), int(A_mouth_box[0][3] - A_mouth_box[0][1]))
    real_A_64_mouth = tf.image.crop_to_bounding_box(A_64, int(A_mouth_box[1][0]), int(A_mouth_box[1][1]), int(A_mouth_box[1][2] - A_mouth_box[1][0]), int(A_mouth_box[1][3] - A_mouth_box[1][1]))
    real_A_32_mouth = tf.image.crop_to_bounding_box(A_32, int(A_mouth_box[2][0]), int(A_mouth_box[2][1]), int(A_mouth_box[2][2] - A_mouth_box[2][0]), int(A_mouth_box[2][3] - A_mouth_box[2][1]))
    real_A_16_mouth = tf.image.crop_to_bounding_box(A_16, int(A_mouth_box[3][0]), int(A_mouth_box[3][1]), int(A_mouth_box[3][2] - A_mouth_box[3][0]), int(A_mouth_box[3][3] - A_mouth_box[3][1]))

    #################################################################################################################################################################################################

    real_B_128_nose = tf.image.crop_to_bounding_box(A_128, int(B_nose_box[0][0]), int(B_nose_box[0][1]), int(B_nose_box[0][2] - B_nose_box[0][0]), int(B_nose_box[0][3] - B_nose_box[0][1]))
    real_B_64_nose = tf.image.crop_to_bounding_box(A_64, int(B_nose_box[1][0]), int(B_nose_box[1][1]), int(B_nose_box[1][2] - B_nose_box[1][0]), int(B_nose_box[1][3] - B_nose_box[1][1]))
    real_B_32_nose = tf.image.crop_to_bounding_box(A_32, int(B_nose_box[2][0]), int(B_nose_box[2][1]), int(B_nose_box[2][2] - B_nose_box[2][0]), int(B_nose_box[2][3] - B_nose_box[2][1]))
    real_B_16_nose = tf.image.crop_to_bounding_box(A_16, int(B_nose_box[3][0]), int(B_nose_box[3][1]), int(B_nose_box[3][2] - B_nose_box[3][0]), int(B_nose_box[3][3] - B_nose_box[3][1]))

    real_B_128_Reyes = tf.image.crop_to_bounding_box(A_128, int(B_eyes_box[0][0][0]), int(B_eyes_box[0][0][1]), int(B_eyes_box[0][0][2] - B_eyes_box[0][0][0]), int(B_eyes_box[0][0][3] - B_eyes_box[0][0][1]))
    real_B_64_Reyes = tf.image.crop_to_bounding_box(A_64, int(B_eyes_box[1][0][0]), int(B_eyes_box[1][0][1]), int(B_eyes_box[1][0][2] - B_eyes_box[1][0][0]), int(B_eyes_box[1][0][3] - B_eyes_box[1][0][1]))
    real_B_32_Reyes = tf.image.crop_to_bounding_box(A_32, int(B_eyes_box[2][0][0]), int(B_eyes_box[2][0][1]), int(B_eyes_box[2][0][2] - B_eyes_box[2][0][0]), int(B_eyes_box[2][0][3] - B_eyes_box[2][0][1]))
    real_B_16_Reyes = tf.image.crop_to_bounding_box(A_16, int(B_eyes_box[3][0][0]), int(B_eyes_box[3][0][1]), int(B_eyes_box[3][0][2] - B_eyes_box[3][0][0]), int(B_eyes_box[3][0][3] - B_eyes_box[3][0][1]))

    real_B_128_Leyes = tf.image.crop_to_bounding_box(A_128, int(B_eyes_box[0][1][0]), int(B_eyes_box[0][1][1]), int(B_eyes_box[0][1][2] - B_eyes_box[0][1][0]), int(B_eyes_box[0][1][3] - B_eyes_box[0][1][1]))
    real_B_64_Leyes = tf.image.crop_to_bounding_box(A_64, int(B_eyes_box[1][1][0]), int(B_eyes_box[1][1][1]), int(B_eyes_box[1][1][2] - B_eyes_box[1][1][0]), int(B_eyes_box[1][1][3] - B_eyes_box[1][1][1]))
    real_B_32_Leyes = tf.image.crop_to_bounding_box(A_32, int(B_eyes_box[2][1][0]), int(B_eyes_box[2][1][1]), int(B_eyes_box[2][1][2] - B_eyes_box[2][1][0]), int(B_eyes_box[2][1][3] - B_eyes_box[2][1][1]))
    real_B_16_Leyes = tf.image.crop_to_bounding_box(A_16, int(B_eyes_box[3][1][0]), int(B_eyes_box[3][1][1]), int(B_eyes_box[3][1][2] - B_eyes_box[3][1][0]), int(B_eyes_box[3][1][3] - B_eyes_box[3][1][1]))

    real_B_128_mouth = tf.image.crop_to_bounding_box(A_128, int(B_mouth_box[0][0]), int(B_mouth_box[0][1]), int(B_mouth_box[0][2] - B_mouth_box[0][0]), int(B_mouth_box[0][3] - B_mouth_box[0][1]))
    real_B_64_mouth = tf.image.crop_to_bounding_box(A_64, int(B_mouth_box[1][0]), int(B_mouth_box[1][1]), int(B_mouth_box[1][2] - B_mouth_box[1][0]), int(B_mouth_box[1][3] - B_mouth_box[1][1]))
    real_B_32_mouth = tf.image.crop_to_bounding_box(A_32, int(B_mouth_box[2][0]), int(B_mouth_box[2][1]), int(B_mouth_box[2][2] - B_mouth_box[2][0]), int(B_mouth_box[2][3] - B_mouth_box[2][1]))
    real_B_16_mouth = tf.image.crop_to_bounding_box(A_16, int(B_mouth_box[3][0]), int(B_mouth_box[3][1]), int(B_mouth_box[3][2] - B_mouth_box[3][0]), int(B_mouth_box[3][3] - B_mouth_box[3][1]))

    real_A_nose_part = (real_A_128_nose, real_A_64_nose, real_A_32_nose, real_A_16_nose)
    real_A_eyes_Rpart = (real_A_128_Reyes, real_A_64_Reyes, real_A_32_Reyes, real_A_16_Reyes)
    real_A_eyes_Lpart = (real_A_128_Leyes, real_A_64_Leyes, real_A_32_Leyes, real_A_16_Leyes)
    real_A_mouth_part = (real_A_128_mouth, real_A_64_mouth, real_A_32_mouth, real_A_16_mouth)
    real_A_all_part = (real_A_nose_part, real_A_eyes_Rpart, real_A_eyes_Lpart, real_A_mouth_part)

    real_B_nose_part = (real_B_128_nose, real_B_64_nose, real_B_32_nose, real_B_16_nose)
    real_B_eyes_Rpart = (real_B_128_Reyes, real_B_64_Reyes, real_B_32_Reyes, real_B_16_Reyes)
    real_B_eyes_Lpart = (real_B_128_Leyes, real_B_64_Leyes, real_B_32_Leyes, real_B_16_Leyes)
    real_B_mouth_part = (real_B_128_mouth, real_B_64_mouth, real_B_32_mouth, real_B_16_mouth)
    real_B_all_part = (real_B_nose_part, real_B_eyes_Rpart, real_B_eyes_Lpart, real_B_mouth_part)

    return real_A_all_part, real_B_all_part

def _func(A_img, B_img):

    A_img = tf.io.read_file(A_img)
    A_img = tf.image.decode_jpeg(A_img, FLAGS.img_ch)
    A_img = tf.image.resize(A_img, [FLAGS.img_size, FLAGS.img_size]) / 127.5 - 1.
    A_gray = tf.image.rgb_to_grayscale(A_img)
    A_p1 = tf.image.resize(A_img, [FLAGS.img_size // 2, FLAGS.img_size // 2])
    A_p1 = tf.image.rgb_to_grayscale(A_p1)

    A_p2 = tf.image.resize(A_img, [FLAGS.img_size // 4, FLAGS.img_size // 4])
    A_p2 = tf.image.rgb_to_grayscale(A_p2)

    A_p3 = tf.image.resize(A_img, [FLAGS.img_size // 8, FLAGS.img_size // 8])
    A_p3 = tf.image.rgb_to_grayscale(A_p3)

    A_p4 = tf.image.resize(A_img, [FLAGS.img_size // 16, FLAGS.img_size // 16])
    A_p4 = tf.image.rgb_to_grayscale(A_p4)

    B_img = tf.io.read_file(B_img)
    B_img = tf.image.decode_jpeg(B_img, FLAGS.img_ch)
    B_img = tf.image.resize(B_img, [FLAGS.img_size, FLAGS.img_size]) / 127.5 - 1.
    B_gray = tf.image.rgb_to_grayscale(B_img)
    B_p1 = tf.image.resize(B_img, [FLAGS.img_size // 2, FLAGS.img_size // 2])
    B_p1 = tf.image.rgb_to_grayscale(B_p1)

    B_p2 = tf.image.resize(B_img, [FLAGS.img_size // 4, FLAGS.img_size // 4])
    B_p2 = tf.image.rgb_to_grayscale(B_p2)

    B_p3 = tf.image.resize(B_img, [FLAGS.img_size // 8, FLAGS.img_size // 8])
    B_p3 = tf.image.rgb_to_grayscale(B_p3)

    B_p4 = tf.image.resize(B_img, [FLAGS.img_size // 16, FLAGS.img_size // 16])
    B_p4 = tf.image.rgb_to_grayscale(B_p4)

    return A_img, (A_gray, A_p1, A_p2, A_p3, A_p4), B_img, (B_gray, B_p1, B_p2, B_p3, B_p4)

#@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(A2B_generator, B2A_generator, A_dis, B_dis,
             A_images, B_images, A_128, A_64, A_32, A_16, B_128, B_64, B_32, B_16,
             real_A_all_part, real_B_all_part,
             A_nose_box, A_eyes_box, A_mouth_box, B_nose_box, B_eyes_box, B_mouth_box):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # 학습이니까 입력을 여러개 넣어야한다. (완료)
        fake_B = run_model(A2B_generator, 
                           [A_images, A_128, A_64, A_32, A_16],
                           True)
        fake_A = run_model(B2A_generator,
                           [B_images, B_128, B_64, B_32, B_16],
                           True)
        fake_B_ = run_model(A2B_generator, 
                            [fake_A[0], A_128, A_64, A_32, A_16],
                            True)
        fake_A_ = run_model(B2A_generator,
                            [fake_B[0], B_128, B_64, B_32, B_16],
                            True)

        DA_part_real, DA_real = run_model(A_dis, A_images, True)
        DB_part_real, DB_real = run_model(B_dis, B_images, True)
        DA_part_fake, DA_fake = run_model(A_dis, fake_A[0], True)
        DB_part_fake, DB_fake = run_model(B_dis, fake_B[0], True)

        fake_B_128_nose = tf.image.crop_to_bounding_box(fake_B[3], int(B_nose_box[0][0]), int(B_nose_box[0][1]), int(B_nose_box[0][2] - B_nose_box[0][0]), int(B_nose_box[0][3] - B_nose_box[0][1]))
        fake_B_64_nose = tf.image.crop_to_bounding_box(fake_B[2], int(B_nose_box[1][0]), int(B_nose_box[1][1]), int(B_nose_box[1][2] - B_nose_box[1][0]), int(B_nose_box[1][3] - B_nose_box[1][1]))
        fake_B_32_nose = tf.image.crop_to_bounding_box(fake_B[1], int(B_nose_box[2][0]), int(B_nose_box[2][1]), int(B_nose_box[2][2] - B_nose_box[2][0]), int(B_nose_box[2][3] - B_nose_box[2][1]))
        A2B_id_nose_loss = tf.reduce_mean(tf.math.abs(fake_B_128_nose - real_B_all_part[0][0])) * 0.8 \
                         + tf.reduce_mean(tf.math.abs(fake_B_64_nose - real_B_all_part[0][1])) * 0.4 \
                         + tf.reduce_mean(tf.math.abs(fake_B_32_nose - real_B_all_part[0][2])) * 0.2
        
        fake_B_128_Reyes = tf.image.crop_to_bounding_box(fake_B[3], int(B_eyes_box[0][0][0]), int(B_eyes_box[0][0][1]), int(B_eyes_box[0][0][2] - B_eyes_box[0][0][0]), int(B_eyes_box[0][0][3] - B_eyes_box[0][0][1]))
        fake_B_128_Leyes = tf.image.crop_to_bounding_box(fake_B[3], int(B_eyes_box[0][1][0]), int(B_eyes_box[0][1][1]), int(B_eyes_box[0][1][2] - B_eyes_box[0][1][0]), int(B_eyes_box[0][1][3] - B_eyes_box[0][1][1]))
        fake_B_64_Reyes = tf.image.crop_to_bounding_box(fake_B[2], int(B_eyes_box[1][0][0]), int(B_eyes_box[1][0][1]), int(B_eyes_box[1][0][2] - B_eyes_box[1][0][0]), int(B_eyes_box[1][0][3] - B_eyes_box[1][0][1]))
        fake_B_64_Leyes = tf.image.crop_to_bounding_box(fake_B[2], int(B_eyes_box[1][1][0]), int(B_eyes_box[1][1][1]), int(B_eyes_box[1][1][2] - B_eyes_box[1][1][0]), int(B_eyes_box[1][1][3] - B_eyes_box[1][1][1]))
        fake_B_32_Reyes = tf.image.crop_to_bounding_box(fake_B[1], int(B_eyes_box[2][0][0]), int(B_eyes_box[2][0][1]), int(B_eyes_box[2][0][2] - B_eyes_box[2][0][0]), int(B_eyes_box[2][0][3] - B_eyes_box[2][0][1]))
        fake_B_32_Leyes = tf.image.crop_to_bounding_box(fake_B[1], int(B_eyes_box[2][1][0]), int(B_eyes_box[2][1][1]), int(B_eyes_box[2][1][2] - B_eyes_box[2][1][0]), int(B_eyes_box[2][1][3] - B_eyes_box[2][1][1]))
        A2B_id_Reyes_loss = tf.reduce_mean(tf.math.abs(fake_B_128_Reyes - real_B_all_part[1][0])) * 0.8 \
                          + tf.reduce_mean(tf.math.abs(fake_B_64_Reyes - real_B_all_part[1][1])) * 0.4 \
                          + tf.reduce_mean(tf.math.abs(fake_B_32_Reyes - real_B_all_part[1][2])) * 0.2
        A2B_id_Leyes_loss = tf.reduce_mean(tf.math.abs(fake_B_128_Leyes - real_B_all_part[2][0])) * 0.8 \
                          + tf.reduce_mean(tf.math.abs(fake_B_64_Leyes - real_B_all_part[2][1])) * 0.4 \
                          + tf.reduce_mean(tf.math.abs(fake_B_32_Leyes - real_B_all_part[2][2])) * 0.2

        fake_B_128_mouth = tf.image.crop_to_bounding_box(fake_B[3], int(B_mouth_box[0][0]), int(B_mouth_box[0][1]), int(B_mouth_box[0][2] - B_mouth_box[0][0]), int(B_mouth_box[0][3] - B_mouth_box[0][1]))
        fake_B_64_mouth = tf.image.crop_to_bounding_box(fake_B[2], int(B_mouth_box[1][0]), int(B_mouth_box[1][1]), int(B_mouth_box[1][2] - B_mouth_box[1][0]), int(B_mouth_box[1][3] - B_mouth_box[1][1]))
        fake_B_32_mouth = tf.image.crop_to_bounding_box(fake_B[1], int(B_mouth_box[2][0]), int(B_mouth_box[2][1]), int(B_mouth_box[2][2] - B_mouth_box[2][0]), int(B_mouth_box[2][3] - B_mouth_box[2][1]))
        A2B_id_mouth_loss = tf.reduce_mean(tf.math.abs(fake_B_128_mouth - real_B_all_part[3][0])) * 0.8 \
                          + tf.reduce_mean(tf.math.abs(fake_B_64_mouth - real_B_all_part[3][1])) * 0.4 \
                          + tf.reduce_mean(tf.math.abs(fake_B_32_mouth - real_B_all_part[3][2])) * 0.2

        ################################################################################################################################################################################################################

        fake_A_128_nose = tf.image.crop_to_bounding_box(fake_A[3], int(A_nose_box[0][0]), int(A_nose_box[0][1]), int(A_nose_box[0][2] - A_nose_box[0][0]), int(A_nose_box[0][3] - A_nose_box[0][1]))
        fake_A_64_nose = tf.image.crop_to_bounding_box(fake_A[2], int(A_nose_box[1][0]), int(A_nose_box[1][1]), int(A_nose_box[1][2] - A_nose_box[1][0]), int(A_nose_box[1][3] - A_nose_box[1][1]))
        fake_A_32_nose = tf.image.crop_to_bounding_box(fake_A[1], int(A_nose_box[2][0]), int(A_nose_box[2][1]), int(A_nose_box[2][2] - A_nose_box[2][0]), int(A_nose_box[2][3] - A_nose_box[2][1]))
        B2A_id_nose_loss = tf.reduce_mean(tf.math.abs(fake_A_128_nose - real_A_all_part[0][0])) * 0.8 \
                         + tf.reduce_mean(tf.math.abs(fake_A_64_nose - real_A_all_part[0][1])) * 0.4 \
                         + tf.reduce_mean(tf.math.abs(fake_A_32_nose - real_A_all_part[0][2])) * 0.2
        
        fake_A_128_Reyes = tf.image.crop_to_bounding_box(fake_A[3], int(A_eyes_box[0][0][0]), int(A_eyes_box[0][0][1]), int(A_eyes_box[0][0][2] - A_eyes_box[0][0][0]), int(A_eyes_box[0][0][3] - A_eyes_box[0][0][1]))
        fake_A_128_Leyes = tf.image.crop_to_bounding_box(fake_A[3], int(A_eyes_box[0][1][0]), int(A_eyes_box[0][1][1]), int(A_eyes_box[0][1][2] - A_eyes_box[0][1][0]), int(A_eyes_box[0][1][3] - A_eyes_box[0][1][1]))
        fake_A_64_Reyes = tf.image.crop_to_bounding_box(fake_A[2], int(A_eyes_box[1][0][0]), int(A_eyes_box[1][0][1]), int(A_eyes_box[1][0][2] - A_eyes_box[1][0][0]), int(A_eyes_box[1][0][3] - A_eyes_box[1][0][1]))
        fake_A_64_Leyes = tf.image.crop_to_bounding_box(fake_A[2], int(A_eyes_box[1][1][0]), int(A_eyes_box[1][1][1]), int(A_eyes_box[1][1][2] - A_eyes_box[1][1][0]), int(A_eyes_box[1][1][3] - A_eyes_box[1][1][1]))
        fake_A_32_Reyes = tf.image.crop_to_bounding_box(fake_A[1], int(A_eyes_box[2][0][0]), int(A_eyes_box[2][0][1]), int(A_eyes_box[2][0][2] - A_eyes_box[2][0][0]), int(A_eyes_box[2][0][3] - A_eyes_box[2][0][1]))
        fake_A_32_Leyes = tf.image.crop_to_bounding_box(fake_A[1], int(A_eyes_box[2][1][0]), int(A_eyes_box[2][1][1]), int(A_eyes_box[2][1][2] - A_eyes_box[2][1][0]), int(A_eyes_box[2][1][3] - A_eyes_box[2][1][1]))
        B2A_id_Reyes_loss = tf.reduce_mean(tf.math.abs(fake_A_128_Reyes - real_A_all_part[1][0])) * 0.8 \
                          + tf.reduce_mean(tf.math.abs(fake_A_64_Reyes - real_A_all_part[1][1])) * 0.4 \
                          + tf.reduce_mean(tf.math.abs(fake_A_32_Reyes - real_A_all_part[1][2])) * 0.2
        B2A_id_Leyes_loss = tf.reduce_mean(tf.math.abs(fake_A_128_Leyes - real_A_all_part[2][0])) * 0.8 \
                          + tf.reduce_mean(tf.math.abs(fake_A_64_Leyes - real_A_all_part[2][1])) * 0.4 \
                          + tf.reduce_mean(tf.math.abs(fake_A_32_Leyes - real_A_all_part[2][2])) * 0.2

        fake_A_128_mouth = tf.image.crop_to_bounding_box(fake_A[3], int(A_mouth_box[0][0]), int(A_mouth_box[0][1]), int(A_mouth_box[0][2] - A_mouth_box[0][0]), int(A_mouth_box[0][3] - A_mouth_box[0][1]))
        fake_A_64_mouth = tf.image.crop_to_bounding_box(fake_A[2], int(A_mouth_box[1][0]), int(A_mouth_box[1][1]), int(A_mouth_box[1][2] - A_mouth_box[1][0]), int(A_mouth_box[1][3] - A_mouth_box[1][1]))
        fake_A_32_mouth = tf.image.crop_to_bounding_box(fake_A[1], int(A_mouth_box[2][0]), int(A_mouth_box[2][1]), int(A_mouth_box[2][2] - A_mouth_box[2][0]), int(A_mouth_box[2][3] - A_mouth_box[2][1]))
        B2A_id_mouth_loss = tf.reduce_mean(tf.math.abs(fake_A_128_mouth - real_A_all_part[3][0])) * 0.8 \
                          + tf.reduce_mean(tf.math.abs(fake_A_64_mouth - real_A_all_part[3][1])) * 0.4 \
                          + tf.reduce_mean(tf.math.abs(fake_A_32_mouth - real_A_all_part[3][2])) * 0.2

        Cycleloss = (tf.reduce_mean(tf.math.abs(fake_A_[0] - A_images)) + tf.reduce_mean(tf.math.abs(fake_B_[0] - B_images)))
        # 이 CycleLOss를 고쳐야 할 것 같다. 왜냐하면
        G_GAN_loss = tf.reduce_mean((DA_part_fake[0] - tf.ones_like(DA_part_fake[0]))**2) \
                   + tf.reduce_mean((DA_part_fake[1] - tf.ones_like(DA_part_fake[1]))**2) \
                   + tf.reduce_mean((DA_part_fake[2] - tf.ones_like(DA_part_fake[2]))**2) \
                   + tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2) \
                   + tf.reduce_mean((DB_part_fake[0] - tf.ones_like(DB_part_fake[0]))**2) \
                   + tf.reduce_mean((DB_part_fake[1] - tf.ones_like(DB_part_fake[1]))**2) \
                   + tf.reduce_mean((DB_part_fake[2] - tf.ones_like(DB_part_fake[2]))**2) \
                   + tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2)

        D_GAN_loss = (tf.reduce_mean((DA_part_fake[0] - tf.zeros_like(DA_part_fake[0]))**2) + tf.reduce_mean((DA_part_real[0] - tf.ones_like(DA_part_real[0]))**2)) * 0.5 \
                   + (tf.reduce_mean((DA_part_fake[1] - tf.zeros_like(DA_part_fake[1]))**2) + tf.reduce_mean((DA_part_real[1] - tf.ones_like(DA_part_real[1]))**2)) * 0.5 \
                   + (tf.reduce_mean((DA_part_fake[2] - tf.zeros_like(DA_part_fake[2]))**2) + tf.reduce_mean((DA_part_real[2] - tf.ones_like(DA_part_real[2]))**2)) * 0.5 \
                   + (tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2) + tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2)) * 0.5 \
                   + (tf.reduce_mean((DB_part_fake[0] - tf.zeros_like(DB_part_fake[0]))**2) + tf.reduce_mean((DB_part_real[0] - tf.ones_like(DB_part_real[0]))**2)) * 0.5 \
                   + (tf.reduce_mean((DB_part_fake[1] - tf.zeros_like(DB_part_fake[1]))**2) + tf.reduce_mean((DB_part_real[1] - tf.ones_like(DB_part_real[1]))**2)) * 0.5 \
                   + (tf.reduce_mean((DB_part_fake[2] - tf.zeros_like(DB_part_fake[2]))**2) + tf.reduce_mean((DB_part_real[2] - tf.ones_like(DB_part_real[2]))**2)) * 0.5 \
                   + (tf.reduce_mean((DB_real - tf.zeros_like(DB_real))**2) + tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2)) * 0.5

        g_loss = G_GAN_loss + Cycleloss + (A2B_id_nose_loss + A2B_id_Reyes_loss + A2B_id_Leyes_loss + A2B_id_mouth_loss) \
                + (B2A_id_nose_loss + B2A_id_Reyes_loss + B2A_id_Leyes_loss + B2A_id_mouth_loss) + tf.reduce_mean(tf.math.abs(fake_B[0] - B_images)) + tf.reduce_mean(tf.math.abs(fake_A[0] - A_images))
        d_loss = D_GAN_loss

    g_grads = g_tape.gradient(g_loss, A2B_generator.trainable_variables + B2A_generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, A_dis.trainable_variables + B_dis.trainable_variables)
    g_optim.apply_gradients(zip(g_grads, A2B_generator.trainable_variables + B2A_generator.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_dis.trainable_variables + B_dis.trainable_variables))
    # 우선은 ID loss를 지금 생각의 흐름대로 정의하였다.
    # 다음으로는 discriminator 를 통한 objective loss를 구해야한다.
    # 그러면 우선적으로 discriminator 의 구조를 어떤식으로 정의해야할지를 정해야한다.
    # 그러므로 내일은 내 모델 discriminator model에 대해 보고 고치자!!!! --> 지금은 discriminator model 고쳐야한다.
    # 잠깐 discriminator를 고치기 전에, facial landmaark를 통해 박스 찾는것을 알아냈다.
    # 박스가 있는곳만 id_loss를 적용하면 괜찮지 않을까??


    return g_loss, d_loss

def main():
    # 모델 정의 위치를 테스트 할 떄와 학습할 때에 각각 정의해주어야한다.
    A2B_generator = New_netowrk_for_generation(input_shape=(FLAGS.img_size, 
                                                            FLAGS.img_size, 
                                                            FLAGS.img_ch))
    B2A_generator = New_netowrk_for_generation(input_shape=(FLAGS.img_size, 
                                                            FLAGS.img_size, 
                                                            FLAGS.img_ch))
    A_dis = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
    B_dis = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))

    A2B_generator.summary()
    A_dis.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_generator=A2B_generator,
                                    B2A_generator=B2A_generator,
                                    g_optim=g_optim,
                                    d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Checkpoint files are restored!!!")

    if FLAGS.train:

        count = 0

        A_img = np.loadtxt(FLAGS.A_tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_img = [FLAGS.A_tr_img_path + img for img in A_img]
        B_img = np.loadtxt(FLAGS.B_tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_img = [FLAGS.B_tr_img_path + img for img in B_img]

        for epoch in range(FLAGS.epochs):

            A = list(zip(A_img, B_img))
            shuffle(A)
            A_img, B_img = zip(*A)
            A_img, B_img = np.array(A_img), np.array(B_img)

            gener = tf.data.Dataset.from_tensor_slices((A_img, B_img))
            gener = gener.shuffle(len(A_img))
            gener = gener.map(_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(gener)
            idx = len(A_img) // FLAGS.batch_size
            for step in range(idx):

                A_imgs, A_img_list, B_imgs, B_img_list = next(it)
                A_value, B_value,\
                A_nose_coord, A_eyes_coord, A_mouth_coord,\
                B_nose_coord, B_eyes_coord, B_mouth_coord,\
                A_nose_box, A_eyes_box, A_mouth_box,\
                B_nose_box, B_eyes_box, B_mouth_box = fasial_lanmark(FLAGS.A_nose_txt,
                                                                    FLAGS.A_eyes_txt,
                                                                    FLAGS.A_mouth_txt,
                                                                    FLAGS.B_nose_txt,
                                                                    FLAGS.B_eyes_txt,
                                                                    FLAGS.B_mouth_txt,
                                                                    A_img_list,
                                                                    B_img_list)

                A_nose_128_coord, A_nose_64_coord, A_nose_32_coord, A_nose_16_coord = A_nose_coord
                A_eye_128_coord, A_eyes_64_coord, A_eyes_32_coord, A_eyes_16_coord = A_eyes_coord
                A_mouth_128_coord, A_mouth_64_coord, A_mouth_32_coord, A_mouth_16_coord = A_mouth_coord

                B_nose_128_coord, B_nose_64_coord, B_nose_32_coord, B_nose_16_coord = B_nose_coord
                B_eye_128_coord, B_eyes_64_coord, B_eyes_32_coord, B_eyes_16_coord = B_eyes_coord
                B_mouth_128_coord, B_mouth_64_coord, B_mouth_32_coord, B_mouth_16_coord = B_mouth_coord

                ########################################################################################################
                # 예시
                #A_128, A_64, A_32, A_16 = A_img_list[1:]
                #a = tf.image.crop_to_bounding_box(A_128, int(A_eyes_box[0][0][0]), int(A_eyes_box[0][0][1]), int(A_eyes_box[0][0][2] - A_eyes_box[0][0][0]), int(A_eyes_box[0][0][3] - A_eyes_box[0][0][1]))
                #plt.imshow(a[0, ..., 0] * 0.5 + 0.5)
                #plt.show()
                ########################################################################################################

                A_128, A_64, A_32, A_16 = A_img_list[1:]
                A_128y, A_128x = tf.image.image_gradients(A_128)
                A_128 = tf.math.add(tf.math.abs(A_128y), tf.math.abs(A_128x))
                A_64y, A_64x = tf.image.image_gradients(A_64)
                A_64 = tf.math.add(tf.math.abs(A_64y), tf.math.abs(A_64x))
                A_32y, A_32x = tf.image.image_gradients(A_32)
                A_32 = tf.math.add(tf.math.abs(A_32y), tf.math.abs(A_32x))
                A_16y, A_16x = tf.image.image_gradients(A_16)
                A_16 = tf.math.add(tf.math.abs(A_16y), tf.math.abs(A_16x))
                A_128, A_64, A_32, A_16 = A_128.numpy(), A_64.numpy(), A_32.numpy(), A_16.numpy()
                te_A_128, te_A_64, te_A_32, te_A_16 = A_128, A_64, A_32, A_16

                B_128, B_64, B_32, B_16 = B_img_list[1:]
                B_128y, B_128x = tf.image.image_gradients(B_128)
                B_128 = tf.math.add(tf.math.abs(B_128y), tf.math.abs(B_128x))
                B_64y, B_64x = tf.image.image_gradients(B_64)
                B_64 = tf.math.add(tf.math.abs(B_64y), tf.math.abs(B_64x))
                B_32y, B_32x = tf.image.image_gradients(B_32)
                B_32 = tf.math.add(tf.math.abs(B_32y), tf.math.abs(B_32x))
                B_16y, B_16x = tf.image.image_gradients(B_16)
                B_16 = tf.math.add(tf.math.abs(B_16y), tf.math.abs(B_16x))
                B_128, B_64, B_32, B_16 = B_128.numpy(), B_64.numpy(), B_32.numpy(), B_16.numpy()
                te_B_128, te_B_64, te_B_32, te_B_16 = B_128, B_64, B_32, B_16

                for j in range(FLAGS.batch_size):
                    for i in range(9):  # nose
                        A_128[:, int(A_nose_128_coord[i, 1]), int(A_nose_128_coord[i, 0]), 0] = A_value[0][j][i]
                        A_64[:, int(A_nose_64_coord[i, 1]), int(A_nose_64_coord[i, 0]), 0] = A_value[0][j][i]
                        A_32[:, int(A_nose_32_coord[i, 1]), int(A_nose_32_coord[i, 0]), 0] = A_value[0][j][i]
                        A_16[:, int(A_nose_16_coord[i, 1]), int(A_nose_16_coord[i, 0]), 0] = A_value[0][j][i]
                        B_128[:, int(B_nose_128_coord[i, 1]), int(B_nose_128_coord[i, 0]), 0] = B_value[0][j][i]
                        B_64[:, int(B_nose_64_coord[i, 1]), int(B_nose_64_coord[i, 0]), 0] = B_value[0][j][i]
                        B_32[:, int(B_nose_32_coord[i, 1]), int(B_nose_32_coord[i, 0]), 0] = B_value[0][j][i]
                        B_16[:, int(B_nose_16_coord[i, 1]), int(B_nose_16_coord[i, 0]), 0] = B_value[0][j][i]
                    for i in range(12): # eyes
                        A_128[:, int(A_eye_128_coord[i, 1]), int(A_eye_128_coord[i, 0]), 0] = A_value[1][j][i]
                        A_64[:, int(A_eyes_64_coord[i, 1]), int(A_eyes_64_coord[i, 0]), 0] = A_value[1][j][i]
                        A_32[:, int(A_eyes_32_coord[i, 1]), int(A_eyes_32_coord[i, 0]), 0] = A_value[1][j][i]
                        A_16[:, int(A_eyes_16_coord[i, 1]), int(A_eyes_16_coord[i, 0]), 0] = A_value[1][j][i]
                        B_128[:, int(B_eye_128_coord[i, 1]), int(B_eye_128_coord[i, 0]), 0] = B_value[1][j][i]
                        B_64[:, int(B_eyes_64_coord[i, 1]), int(B_eyes_64_coord[i, 0]), 0] = B_value[1][j][i]
                        B_32[:, int(B_eyes_32_coord[i, 1]), int(B_eyes_32_coord[i, 0]), 0] = B_value[1][j][i]
                        B_16[:, int(B_eyes_16_coord[i, 1]), int(B_eyes_16_coord[i, 0]), 0] = B_value[1][j][i]
                    for i in range(20): # mouth
                        A_128[:, int(A_mouth_128_coord[i, 1]), int(A_mouth_128_coord[i, 0]), 0] = A_value[2][j][i]
                        A_64[:, int(A_mouth_64_coord[i, 1]), int(A_mouth_64_coord[i, 0]), 0] = A_value[2][j][i]
                        A_32[:, int(A_mouth_32_coord[i, 1]), int(A_mouth_32_coord[i, 0]), 0] = A_value[2][j][i]
                        A_16[:, int(A_mouth_16_coord[i, 1]), int(A_mouth_16_coord[i, 0]), 0] = A_value[2][j][i]
                        B_128[:, int(B_mouth_128_coord[i, 1]), int(B_mouth_128_coord[i, 0]), 0] = B_value[2][j][i]
                        B_64[:, int(B_mouth_64_coord[i, 1]), int(B_mouth_64_coord[i, 0]), 0] = B_value[2][j][i]
                        B_32[:, int(B_mouth_32_coord[i, 1]), int(B_mouth_32_coord[i, 0]), 0] = B_value[2][j][i]
                        B_16[:, int(B_mouth_16_coord[i, 1]), int(B_mouth_16_coord[i, 0]), 0] = B_value[2][j][i]


                real_A_all_part, real_B_all_part = crop_image_part(A_nose_box, A_eyes_box, A_mouth_box, B_nose_box, B_eyes_box, B_mouth_box,
                                                                   A_128, A_64, A_32, A_16, B_128, B_64, B_32, B_16)
                g_loss, d_loss = cal_loss(A2B_generator, 
                                B2A_generator, 
                                A_dis, 
                                B_dis, 
                                A_imgs, 
                                B_imgs, 
                                A_128, A_64, A_32, A_16, B_128, B_64, B_32, B_16,
                                real_A_all_part,
                                real_B_all_part,
                                A_nose_box, A_eyes_box, A_mouth_box, B_nose_box, B_eyes_box, B_mouth_box)

                print(g_loss, d_loss, count)

                if count % 100 == 0:
                    # 학습은 되고 있는데, 이미지가 변하지 않는건.. 아마 model을 각각 따로 써줘서 그럴수도있다.
                    # 그냥 학습과 똑같이 입력을 여러개로 진행할까?? 한번 생각해보자! 내일꼭!!!! (지금으로서는 거의 80%가 모델을 따로줘서 학습된 웨이트가 제대로 안들어가는것같다)
                    # 이렇게 되면 generator 모델안에 trainable 에 대한 인퍼런스를 제거해야한다.
                    # 입력을 매번 이런식으로...????
                    fake_B = run_model(A2B_generator, [A_imgs, te_A_128, te_A_64, te_A_32, te_A_16], False) # 우선은 실험만 해보자
                    fake_A = run_model(B2A_generator, [B_imgs, te_B_128, te_B_64, te_B_32, te_B_16], False)

                    plt.imsave("C:/Users/Yuhwan/Pictures/sample_images/fake_B_{}.png".format(count), fake_B[0][0].numpy() * 0.5 + 0.5)
                    plt.imsave("C:/Users/Yuhwan/Pictures/sample_images/fake_A_{}.png".format(count), fake_A[0][0].numpy() * 0.5 + 0.5)
                    plt.imsave("C:/Users/Yuhwan/Pictures/sample_images/real_A_{}.png".format(count), A_imgs[0].numpy() * 0.5 + 0.5)
                    plt.imsave("C:/Users/Yuhwan/Pictures/sample_images/real_B_{}.png".format(count), B_imgs[0].numpy() * 0.5 + 0.5)

                count += 1

if __name__ == "__main__":
    main()
