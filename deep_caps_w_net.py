from caspule_layers import *
import tensorflow as tf
import numpy as np
import cv2

def cap_U_encoder(input, K):
    x              = tf.reshape(input, shape=[-1, 224, 224, 3])

    conv1          = tf.layers.conv2d(x, 16, 5, 1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    print("Conv1_Shape : ", conv1.get_shape())
    conv1_expanded = tf.expand_dims(conv1, axis=3)
    #print("Conv1_Expanded Shape : ", conv1_expanded.get_shape())

    conv_caps_1    = SegConvolution(conv1_expanded, 5, 2, 16, strides=2, routings=1)
    print("Conv Caps 1 Shape : ", conv_caps_1.get_shape())
    conv_caps_2    = SegConvolution(conv_caps_1, 5, 4, 16, strides=1, routings=3)
    print("Conv Caps 2 Shape : ", conv_caps_2.get_shape())

    conv_caps_3    = SegConvolution(conv_caps_2, 5, 4, 32, strides=2, routings=3)
    print("Conv Caps 3 Shape : ", conv_caps_3.get_shape())
    conv_caps_4    = SegConvolution(conv_caps_3, 5, 8, 32, strides=1, routings=3)
    print("Conv Caps 4 Shape : ", conv_caps_4.get_shape())

    conv_caps_5    = SegConvolution(conv_caps_4, 5, 8, 64, strides=2, routings=3)
    print("Conv Caps 5 Shape : ", conv_caps_5.get_shape())
    conv_caps_6    = SegConvolution(conv_caps_5, 5, 8, 64, strides=1, routings=3)
    print("Conv Caps 6 Shape : ", conv_caps_6.get_shape())

    conv_caps_7    = SegConvolution(conv_caps_6, 5, 8, 128, strides=2, routings=3)
    print("Conv Caps 7 Shape : ", conv_caps_7.get_shape())
    conv_caps_8    = SegConvolution(conv_caps_7, 5, 16, 64, strides=1, routings=3)
    print("Conv Caps 8 Shape : ", conv_caps_8.get_shape())

    deconv_caps_1  = SegConvolution(conv_caps_8, 4, 16, 64, strides=2, routings=3, op='deconv')
    print("Deconv 1 Shape : ", deconv_caps_1.get_shape())
    concat_1       = tf.concat([deconv_caps_1, conv_caps_6], axis=3)
    conv_caps_9    = SegConvolution(concat_1, 5, 8, 32, strides=1, routings=3)
    print("Conv Caps 9 Shape : ", conv_caps_9.get_shape())

    deconv_caps_2  = SegConvolution(conv_caps_9, 4, 8, 32, strides=2, routings=3, op='deconv')
    print("Deconv 2 Shape : ", deconv_caps_2.get_shape())
    concat_2       = tf.concat([deconv_caps_2, conv_caps_4], axis=3)
    conv_caps_10   = SegConvolution(concat_2, 5, 4, 16, strides=1, routings=3)
    print("Conv Caps 10 Shape : ", conv_caps_10.get_shape())

    deconv_caps_3  = SegConvolution(conv_caps_10, 4, 4, 16, strides=2, routings=3, op='deconv')
    print("Deconv 2 Shape : ", deconv_caps_2.get_shape())
    concat_3       = tf.concat([deconv_caps_3, conv_caps_2], axis=3)
    conv_caps_11   = SegConvolution(concat_3, 5, 2, 16, strides=1, routings=3)
    print("Conv Caps 11 Shape : ", conv_caps_11.get_shape())

    deconv_caps_4  = SegConvolution(conv_caps_11, 4, 2, 16, strides=2, routings=3, op='deconv')
    print("Deconv 4 Shape : ", deconv_caps_4.get_shape())
    concat_4       = tf.concat([deconv_caps_4, conv1_expanded], axis=3)
    conv_caps_12   = SegConvolution(concat_4, 1, 1, K, strides=1, routings=3)
    print("Conv Caps 12 Shape : ", conv_caps_12.get_shape())

    squeeze        = tf.squeeze(conv_caps_12, axis=3)
    #print("Squeeze Shape : ", squeeze.get_shape())
    softmax        = softmax_1 = tf.nn.softmax(squeeze)
    print("Encode Output Shape : " , softmax.get_shape())

    return softmax

def cap_U_decoder(input):
    dec_conv1          = tf.layers.conv2d(input, 16, 5, 1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    print("Conv1_Shape : ", dec_conv1.get_shape())
    dec_conv1_expanded = tf.expand_dims(dec_conv1, axis=3)
    #print("Conv1_Expanded Shape : ", conv1_expanded.get_shape())

    dec_conv_caps_1    = SegConvolution(dec_conv1_expanded, 5, 2, 16, strides=2, routings=1)
    print("Conv Caps 1 Shape : ", dec_conv_caps_1.get_shape())
    dec_conv_caps_2    = SegConvolution(dec_conv_caps_1, 5, 4, 16, strides=1, routings=3)
    print("Conv Caps 2 Shape : ", dec_conv_caps_2.get_shape())

    dec_conv_caps_3    = SegConvolution(dec_conv_caps_2, 5, 4, 32, strides=2, routings=3)
    print("Conv Caps 3 Shape : ", dec_conv_caps_3.get_shape())
    dec_conv_caps_4    = SegConvolution(dec_conv_caps_3, 5, 8, 32, strides=1, routings=3)
    print("Conv Caps 4 Shape : ", dec_conv_caps_4.get_shape())

    dec_conv_caps_5    = SegConvolution(dec_conv_caps_4, 5, 8, 64, strides=2, routings=3)
    print("Conv Caps 5 Shape : ", dec_conv_caps_5.get_shape())
    dec_conv_caps_6    = SegConvolution(dec_conv_caps_5, 5, 8, 64, strides=1, routings=3)
    print("Conv Caps 6 Shape : ", dec_conv_caps_6.get_shape())

    dec_conv_caps_7    = SegConvolution(dec_conv_caps_6, 5, 8, 128, strides=2, routings=3)
    print("Conv Caps 7 Shape : ", dec_conv_caps_7.get_shape())
    dec_conv_caps_8    = SegConvolution(dec_conv_caps_7, 5, 16, 64, strides=1, routings=3)
    print("Conv Caps 8 Shape : ", dec_conv_caps_8.get_shape())

    dec_deconv_caps_1  = SegConvolution(dec_conv_caps_8, 4, 16, 64, strides=2, routings=3, op='deconv')
    print("Deconv 1 Shape : ", dec_deconv_caps_1.get_shape())
    dec_concat_1       = tf.concat([dec_deconv_caps_1, dec_conv_caps_6], axis=3)
    dec_conv_caps_9    = SegConvolution(dec_concat_1, 5, 8, 32, strides=1, routings=3)
    print("Conv Caps 9 Shape : ", dec_conv_caps_9.get_shape())

    dec_deconv_caps_2  = SegConvolution(dec_conv_caps_9, 4, 8, 32, strides=2, routings=3, op='deconv')
    print("Deconv 2 Shape : ", dec_deconv_caps_2.get_shape())
    dec_concat_2       = tf.concat([dec_deconv_caps_2, dec_conv_caps_4], axis=3)
    dec_conv_caps_10   = SegConvolution(dec_concat_2, 5, 4, 16, strides=1, routings=3)
    print("Conv Caps 10 Shape : ", dec_conv_caps_10.get_shape())

    dec_deconv_caps_3  = SegConvolution(dec_conv_caps_10, 4, 4, 16, strides=2, routings=3, op='deconv')
    print("Deconv 3 Shape : ", dec_deconv_caps_3.get_shape())
    dec_concat_3       = tf.concat([dec_deconv_caps_3, dec_conv_caps_2], axis=3)
    dec_conv_caps_11   = SegConvolution(dec_concat_3, 5, 2, 16, strides=1, routings=3)
    print("Conv Caps 11 Shape : ", dec_conv_caps_11.get_shape())

    dec_deconv_caps_4  = SegConvolution(dec_conv_caps_11, 4, 2, 16, strides=2, routings=3, op='deconv')
    print("Deconv 3 Shape : ", dec_deconv_caps_4.get_shape())
    dec_concat_4       = tf.concat([dec_deconv_caps_4, dec_conv1_expanded], axis=3)
    dec_conv_caps_12   = SegConvolution(dec_concat_4, 1, 1, 16, strides=1, routings=3)
    print("Conv Caps 12 Shape : ", dec_conv_caps_12.get_shape())

    dec_squeeze        = tf.squeeze(dec_conv_caps_12, axis=3)

    #Reconstruction
    rec_conv_1      = tf.layers.conv2d(dec_squeeze, 64, 1, 1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    rec_conv_2      = tf.layers.conv2d(rec_conv_1, 128, 1, 1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output          = tf.layers.conv2d(rec_conv_2, 3, kernel_size = 1, padding="same")

    print("Decoder Output Shape : " , output.get_shape())

    return output

if __name__ == "__main__":
    batch_size = 1
    K = 4
    width = 224
    height = 224

    img     = cv2.imread("gdz.png")
    img     = cv2.resize(img, (width, height))
    img     = np.array([img])

    input   = np.random.uniform(size=(1, width, height, 3))
    y       = np.random.uniform(size=(1, width, height, 3))

    image   = tf.placeholder(tf.float32, [None, width, height, 3])
    segment = tf.placeholder(tf.float32, [None, width, height, 3])

    encoder = cap_U_encoder(image, K)
    decoder = cap_U_decoder(encoder)
