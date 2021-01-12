# -*- coding:utf-8 -*-
import tensorflow as tf

# 얼굴의 고주파 성분을 감지하여 생성되는 결과물에 대해 좀더 명확한 이미지를 얻을 수 있는 네트워크를 만들어보자
# FPN에서 했던것을 응용해보자!! 기억해!! backbone model은 mobilenet V2 (약간 수정하자) 로 선정

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class Down_input(tf.keras.layers.Layer):
    def __init__(self):
        super(Down_input, self).__init__()

    def call(self, layer_input, img_input):

        x = tf.keras.layers.Add()([layer_input, img_input])
        #x = layer_input * img_input

        return x

def New_netowrk_for_generation(input_shape=(256, 256, 3), trainable=True):
    
    def residual_block(input, filters):

        h = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False)(input)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=filters*2,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False)(h)
        h = InstanceNormalization()(h)

        return tf.keras.layers.ReLU()(h + input)

    h = inputs = tf.keras.Input(input_shape)
    if trainable:
        down_input1 = tf.keras.Input((input_shape[0] // 2,
                                       input_shape[1] // 2,
                                       1))  # [128, 128]
        down_input2 = tf.keras.Input((input_shape[0] // 4,
                                      input_shape[1] // 4,
                                      1))   # [64, 64]
        down_input3 = tf.keras.Input((input_shape[0] // 8,
                                      input_shape[1] // 8,
                                      1))   # [32, 32]
        down_input4 = tf.keras.Input((input_shape[0] // 16,
                                      input_shape[1] // 16,
                                      1))   # [16, 16]
    # Encode part
    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 32]

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 64]
    if trainable:
        h = Down_input()(h, down_input1)
    for _ in range(1):
        h = residual_block(h, 32)   # [128, 128, 64]
    C1 = h

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 128]
    if trainable:
        h = Down_input()(h, down_input2)
    for _ in range(2):
        h = residual_block(h, 64)   # [64, 64, 128]
    C2 = h
    
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [32, 32, 256]
    if trainable:
        h = Down_input()(h, down_input3)
    for _ in range(4):
        h = residual_block(h, 128)  # [32, 32, 256]
    C3 = h

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [16, 16, 512]
    if trainable:
        h = Down_input()(h, down_input4)
    for _ in range(3):
        h = residual_block(h, 256) # [16, 16, 512]

    C4 = h
    P4 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=1,
                                strides=1,
                                padding="same")(C4)  # [16, 16, 256]
    h = tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [32, 32, 256]

    h1 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P4),
                               h]) # [32, 32, 256]

    P3 = tf.keras.layers.Conv2D(filters=128,
                                kernel_size=1,
                                strides=1,
                                padding="same")(C3) # [32, 32, 128]

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False)(h1)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 128]

    h2 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P3),
                               h])  # [64, 64, 128]

    P2 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=1,
                                strides=1,
                                padding="same")(C2) # [64, 64, 64]

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False)(h2)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 64]

    h3 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P2),
                               h])  # [128, 128, 64]

    P1 = tf.keras.layers.Conv2D(filters=32,
                                kernel_size=1,
                                strides=1,
                                padding="same")(C1) # [128, 128, 32]

    h = tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False)(h3)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 32]

    h4 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2,2))(P1),
                               h])  # [256, 256, 32]

    h = tf.keras.layers.ZeroPadding2D((3,3))(h4)
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid")(h)  # [256, 256, 3]
    h = tf.keras.layers.Activation("tanh")(h)
    
    if trainable:
        return tf.keras.Model(inputs=[inputs, 
                                      down_input1,
                                      down_input2,
                                      down_input3,
                                      down_input4], 
                              outputs=[h, h1, h2, h3, h4])
    else:
        return tf.keras.Model(inputs=inputs, 
                              outputs=[h, h1, h2, h3, h4])

def Discriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    # 이제 여기를 고쳐야한다.!!!!

    dim_ = dim

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same')(h)    # [256, 256, 64]
    P1 = h
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(dim*2, 4, strides=2, padding='same')(h)    # [128, 128, 128]
    P2 = tf.keras.layers.Conv2D(filters=3,
                                kernel_size=4,
                                strides=1,
                                padding="same")(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    dim = min(dim * 4, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
    P3 = tf.keras.layers.Conv2D(filters=3,
                                kernel_size=4,
                                strides=1,
                                padding="same")(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [64, 64, 256]

    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
    P4 = tf.keras.layers.Conv2D(filters=3,
                                kernel_size=4,
                                strides=1,
                                padding="same")(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [32, 32, 256]

    # 2
    dim = min(dim * 8, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h) # [32, 32, 512]

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    output = [P2, P3, P4]
    final_feature = h

    return tf.keras.Model(inputs=inputs, outputs=[output, final_feature])

model = New_netowrk_for_generation(trainable=True)
model.summary()