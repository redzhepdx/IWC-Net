from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from model import *

tf.logging.set_verbosity(tf.logging.INFO)

def create_model(features, labels, mode, params):
    encoded     = U_encoder(features["images"], params["K"])
    decoded     = U_decoder(encoded)

    predictions = {"reconstructed_image" : decoded,
                   "segmentation" : tf.argmax(encoded, axis=3)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    norm_loss   = ncut_loss_tf(encoded, features["images"], params["positions"], params["pos_distances"], params["K"], params["batch_size"])
    loss        = min_square_error(decoded, labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("TRAINING PHASE")
        optimizer                = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op_norm_cut        = optimizer.minimize(loss=norm_loss)
        train_op_reconstruction  = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        train_op_merged          = tf.group(train_op_norm_cut, train_op_reconstruction)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss + norm_loss, train_op=train_op_merged)

    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["reconstructed_image"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(_):
    K            = 4
    width        = 224
    height       = 224
    batch_size   = 1
    dataset_size = 20

    #Data Preparation
    sparse_matrix = positional_sparse_matrix(width, height, 5)
    positions     = np.array(list(zip(sparse_matrix.nonzero()[0], sparse_matrix.nonzero()[1])))
    pos_distances = np.reshape(sparse_matrix.data[sparse_matrix.data > 0], (-1, 1))
    train_data    = np.random.randint(255, size=(dataset_size, width, height, 3)) ## TODO : Load Dataset Instead of Random
    #Model Training Setup
    w_net_segmentator = tf.estimator.Estimator(model_fn=create_model,
                                               model_dir="/models/w_net_model",
                                               params = {"positions" : positions,
                                                         "pos_distances" : pos_distances,
                                                         "batch_size" : batch_size,
                                                         "K": K})
    train_input_fn    = tf.estimator.inputs.numpy_input_fn(
        x={"images": train_data.astype(np.float32)},
        y=train_data.astype(np.float32), batch_size=batch_size, num_epochs=None, shuffle=True)

    #Boom Train the model
    w_net_segmentator.train(input_fn = train_input_fn, steps=50000)

if __name__ == "__main__":
  tf.app.run()
