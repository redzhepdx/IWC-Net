import tensorflow as tf
import numpy as np

def squash(input_tensor):
    norm    = tf.norm(input_tensor, axis=-1, keep_dims=True)
    norm_sq = tf.square(norm)
    res     = tf.multiply(tf.div(input_tensor, norm), tf.div(norm_sq, 1 + norm_sq))
    return res

def squash2(p):
    p_norm_sq = tf.reduce_sum(tf.square(p), axis=-1, keep_dims=True)
    p_norm = tf.sqrt(p_norm_sq + 1e-9)
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v

def compute_vector_length(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True) + 1e-9)


def SegConvolution(input_tensor, kernel_size, num_capsules, num_atoms, strides=1, padding='same', routings=3, op='conv'):
    '''
    input_tensor : tensor with shape -> (N, H, W, capsule_prev, atom_prev)
    kernel_size  : (k, k)
    num_capsule  : next_capsule capsule count
    num_atomns   : depth(dimensions) of capule
    routings     : number of routings
    op           : operation type = [convolution, deconvoluition(transposed_convolution), separable_convolution]
    '''

    shape      = input_tensor.get_shape()
    batch_size = tf.shape(input_tensor)[0]
    caps_prev  = int(shape[3])
    atom_prev  = int(shape[4])

    #Slice The Capsule
    caps_slice_list = [tf.squeeze(caps_slice, axis=3) for caps_slice in tf.split(input_tensor, caps_prev, axis=3)]
    conv_caps_list  = []

    for caps_slice in caps_slice_list:
        if op == 'conv':
            conv_slice = tf.layers.conv2d(caps_slice, num_capsules * num_atoms, kernel_size = kernel_size, strides=strides, padding=padding, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        elif op == 'sep_conv':
            conv_slice = tf.layers.separable_conv2d(caps_slice,  num_capsules * num_atoms, kernel_size = kernel_size, depth_multiplier = 1, padding=padding, depthwise_initializer=tf.truncated_normal_initializer(stddev=0.01))
            #TODO In progress
        elif op == 'deconv':
            conv_slice = tf.layers.conv2d_transpose(caps_slice,  num_capsules * num_atoms, kernel_size = kernel_size, strides=strides, padding=padding, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        else:
            raise ValueError("Check Your Type Operation , Something Wrong!!!")

        conv_shape  = conv_slice.get_shape()
        conv_height = int(conv_shape[1])
        conv_width  = int(conv_shape[2])

        #Convert to capsule format
        conv_slice  = tf.reshape(conv_slice, [batch_size, conv_height, conv_width, num_capsules, num_atoms])

        conv_caps_list.append(conv_slice)

    one_kernel = tf.ones([kernel_size, kernel_size, num_capsules, 1])
    bias_zero  = tf.zeros([batch_size, conv_height, conv_width, caps_prev, num_capsules])

    bias_slice_list   = [tf.squeeze(bias_slice, axis=3) for bias_slice in tf.split(bias_zero, caps_prev, axis=3)]
    #stop_gradient
    conv_caps_list_sq = [tf.stop_gradient(conv_caps) for conv_caps in conv_caps_list]

    for d in range(routings):
        if d < (routings - 1):
            conv_caps_list_prime = conv_caps_list_sq
        else:
            conv_caps_list_prime = conv_caps_list

        mul_conv_caps_list = []
        for bias_tensor, conv_caps_tensor in zip(bias_slice_list, conv_caps_list_prime):
            bias_tensor_max   = tf.nn.max_pool(bias_tensor, [1, kernel_size, kernel_size, 1], [1, 1, 1, 1], "SAME")
            bias_tensor_max   = tf.reduce_max(bias_tensor_max, axis=3, keep_dims=True)

            diff_exp_bias     = tf.exp(tf.subtract(bias_tensor, bias_tensor_max))
            sum_diff_exp_bias = tf.nn.conv2d(diff_exp_bias, one_kernel, [1,1,1,1], "SAME")

            route_result      = tf.div(diff_exp_bias, sum_diff_exp_bias)
            route_result      = tf.expand_dims(route_result, axis=4)
            mul_conv_caps_list.append(tf.multiply(route_result, conv_caps_tensor)) #TODO It can be MATMUL

        p = tf.add_n(mul_conv_caps_list)
        v = squash2(p)

        if d < (routings - 1):
            bias_tensor_list_prime = []
            for bias_tensor, conv_caps_tensor in zip(bias_slice_list, conv_caps_list_prime):
                res = tf.add(bias_tensor, tf.reduce_sum(tf.multiply(conv_caps_tensor, v), axis=4))
                bias_tensor_list_prime.append(res)
            bias_slice_list = bias_tensor_list_prime

    return v

def InceptionCapsuleLayer(input_tensor, capsules, atoms):
    conv_caps_5x5_1    = SegConvolution(input_tensor, 1, capsules, atoms, strides=1, routings=3)
    conv_caps_5x5_2    = SegConvolution(conv_caps_5x5_1, 5, capsules, atoms, strides=1, routings=3)

    conv_caps_3x3_1    = SegConvolution(input_tensor, 1, capsules, atoms, strides=1, routings=3)
    conv_caps_3x3_2    = SegConvolution(conv_caps_3x3_1, 3, capsules, atoms, strides=1, routings=3)

    conv_caps_1x1_1    = SegConvolution(input_tensor, 1, capsules, atoms, strides=1, routings=3)

    inception = tf.concat([conv_caps_3x3_2, conv_caps_5x5_2, conv_caps_1x1_1], axis=3)

    return inception
