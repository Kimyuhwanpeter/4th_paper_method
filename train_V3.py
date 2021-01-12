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

flags.DEFINE_integer("batch_size", 1, "")

flags.DEFINE_integer("epochs", 200, "")

flags.DEFINE_float("lr", 0.0002, "")

flags.DEFINE_bool("train", True, "")

flags.DEFINE_bool("pre_checkpoint", False, "")

flags.DEFINE_string("pre_checkpoint_path", "", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

g_optim = tf.keras.optimizers.Adam(FLAGS.lr)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr)

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

    # 읽은 좌표값에 대해 rescale도 따로 해주어야한다. (각 레이어에 넣어줘야 하기 떄문이다.)
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

    # tensorflow 에서는 h,w 이고 opencv에서는 w,h 이다.
    # 그렇기 떄문에 여기서는 x,y (w, h) --> y,x (h, w)로 바꿔서 적용해야 한다.

    # 각 입력의 원본 이미지에 대한 좌표에 해당하는 값을 각 레이어에 반영해주어야한다.
    # 입력이미지의 좌표에 해당하는 값을 가지고 와야한다.
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

    #return (A_value, A_nose_128, A_nose_64, A_nose_32, A_nose_16), (B_value, B_nose_128, B_nose_64, B_nose_32, B_nose_16)
    return A_value, B_value, A_nose_coord, A_eyes_coord, A_mouth_coord, B_nose_coord, B_eyes_coord, B_mouth_coord

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

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(A2B_generator, B2A_generator, A_dis, B_dis,
             A_images, B_images, A_128, A_64, A_32, A_16, B_128, B_64, B_32, B_16):

    with tf.GradientTape() as tape:
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

        id_loss = 2 * tf.reduce_mean(tf.math.abs(fake_B[1] - A_128)) \
            + 4 * tf.reduce_mean(tf.math.abs(fake_B[2] - A_64)) \
            + 6 * tf.reduce_mean(tf.math.abs(fake_B[3] - A_32)) \
            + 8 * tf.reduce_mean(tf.math.abs(fake_B[4] - A_16)) \
            + 2 * tf.reduce_mean(tf.math.abs(fake_A[1] - B_128)) \
            + 4 * tf.reduce_mean(tf.math.abs(fake_A[2] - B_64)) \
            + 6 * tf.reduce_mean(tf.math.abs(fake_A[3] - B_32)) \
            + 8 * tf.reduce_mean(tf.math.abs(fake_A[4] - B_16))
        
        # 우선은 ID loss를 지금 생각의 흐름대로 정의하였다.
        # 다음으로는 discriminator 를 통한 objective loss를 구해야한다.
        # 그러면 우선적으로 discriminator 의 구조를 어떤식으로 정의해야할지를 정해야한다.
        # 그러므로 내일은 내 모델 discriminator model에 대해 보고 고치자!!!! --> 지금은 discriminator model 고쳐야한다.

    return loss

def main():
    # 모델 정의 위치를 테스트 할 떄와 학습할 때에 각각 정의해주어야한다.

    if FLAGS.train:

        A2B_generator = New_netowrk_for_generation(input_shape=(FLAGS.img_size, 
                                                                FLAGS.img_size, 
                                                                FLAGS.img_ch),
                                                   trainable=True)
        B2A_generator = New_netowrk_for_generation(input_shape=(FLAGS.img_size, 
                                                                FLAGS.img_size, 
                                                                FLAGS.img_ch),
                                                   trainable=True)
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
                A_value, B_value, A_nose_coord, A_eyes_coord, A_mouth_coord, B_nose_coord, B_eyes_coord, B_mouth_coord = fasial_lanmark(FLAGS.A_nose_txt,
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

                for j in range(FLAGS.batch_size):
                    for i in range(9):
                        A_128[:, int(A_nose_128_coord[i, 1]), int(A_nose_128_coord[i, 0]), 0] = A_value[0][j][i]
                        A_64[:, int(A_nose_64_coord[i, 1]), int(A_nose_64_coord[i, 0]), 0] = A_value[0][j][i]
                        A_32[:, int(A_nose_32_coord[i, 1]), int(A_nose_32_coord[i, 0]), 0] = A_value[0][j][i]
                        A_16[:, int(A_nose_16_coord[i, 1]), int(A_nose_16_coord[i, 0]), 0] = A_value[0][j][i]
                        B_128[:, int(B_nose_128_coord[i, 1]), int(B_nose_128_coord[i, 0]), 0] = B_value[0][j][i]
                        B_64[:, int(B_nose_64_coord[i, 1]), int(B_nose_64_coord[i, 0]), 0] = B_value[0][j][i]
                        B_32[:, int(B_nose_32_coord[i, 1]), int(B_nose_32_coord[i, 0]), 0] = B_value[0][j][i]
                        B_16[:, int(B_nose_16_coord[i, 1]), int(B_nose_16_coord[i, 0]), 0] = B_value[0][j][i]
                    for i in range(12):
                        A_128[:, int(A_eye_128_coord[i, 1]), int(A_eye_128_coord[i, 0]), 0] = A_value[1][j][i]
                        A_64[:, int(A_eyes_64_coord[i, 1]), int(A_eyes_64_coord[i, 0]), 0] = A_value[1][j][i]
                        A_32[:, int(A_eyes_32_coord[i, 1]), int(A_eyes_32_coord[i, 0]), 0] = A_value[1][j][i]
                        A_16[:, int(A_eyes_16_coord[i, 1]), int(A_eyes_16_coord[i, 0]), 0] = A_value[1][j][i]
                        B_128[:, int(B_eye_128_coord[i, 1]), int(B_eye_128_coord[i, 0]), 0] = B_value[1][j][i]
                        B_64[:, int(B_eyes_64_coord[i, 1]), int(B_eyes_64_coord[i, 0]), 0] = B_value[1][j][i]
                        B_32[:, int(B_eyes_32_coord[i, 1]), int(B_eyes_32_coord[i, 0]), 0] = B_value[1][j][i]
                        B_16[:, int(B_eyes_16_coord[i, 1]), int(B_eyes_16_coord[i, 0]), 0] = B_value[1][j][i]
                    for i in range(20):
                        A_128[:, int(A_mouth_128_coord[i, 1]), int(A_mouth_128_coord[i, 0]), 0] = A_value[2][j][i]
                        A_64[:, int(A_mouth_64_coord[i, 1]), int(A_mouth_64_coord[i, 0]), 0] = A_value[2][j][i]
                        A_32[:, int(A_mouth_32_coord[i, 1]), int(A_mouth_32_coord[i, 0]), 0] = A_value[2][j][i]
                        A_16[:, int(A_mouth_16_coord[i, 1]), int(A_mouth_16_coord[i, 0]), 0] = A_value[2][j][i]
                        B_128[:, int(B_mouth_128_coord[i, 1]), int(B_mouth_128_coord[i, 0]), 0] = B_value[2][j][i]
                        B_64[:, int(B_mouth_64_coord[i, 1]), int(B_mouth_64_coord[i, 0]), 0] = B_value[2][j][i]
                        B_32[:, int(B_mouth_32_coord[i, 1]), int(B_mouth_32_coord[i, 0]), 0] = B_value[2][j][i]
                        B_16[:, int(B_mouth_16_coord[i, 1]), int(B_mouth_16_coord[i, 0]), 0] = B_value[2][j][i]


                loss = cal_loss(A2B_generator, 
                                B2A_generator, 
                                A_dis, 
                                B_dis, 
                                A_imgs, 
                                B_imgs, 
                                A_128, A_64, A_32, A_16, B_128, B_64, B_32, B_16)

                print(step)

if __name__ == "__main__":
    main()