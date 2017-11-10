from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import urllib

import tensorflow as tf
import os
# need it for some werid import error
from pandas.core.computation import expressions
import numpy as np

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('--model_dir', './model', 'folder with model checkpoint file')
tf.flags.DEFINE_string('--export_dir', './exported_model', 'folder to export to')
tf.flags.DEFINE_string('--version', 1, 'exported model version')

MODEL_DIR = FLAGS.model_dir
VERSION = FLAGS.version
EXPORT_DIR = os.path.join(FLAGS.export_dir, '%i/' % VERSION)
N_CLASSES = 2

## build data pipe - I find using the dataset on generator quite 
## universal for many cases. Here the data is artifical for demos.

def get_input_fn(dataset, batch_size = 32, repeats=1, shuffle=False):
    def _data_generator():
        for i in range(len(dataset)):
            x, y = dataset[i]
            yield (x, np.int32(y))
    def _input_fn():
        tf_dataset = tf.data.Dataset.from_generator(
            _data_generator, 
            output_types=(tf.float32, tf.int32),
            output_shapes=(tf.TensorShape([300, 3]), tf.TensorShape([])) )
        tf_dataset = tf_dataset.repeat(repeats).batch(batch_size)
        if shuffle:
            tf_dataset = tf_dataset.shuffle(batch_size * 5)
        iterator = tf_dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return (batch_x, batch_y)
    return _input_fn

## Build the model
def model_fn(mode, features, labels):
    ## see discussion here https://github.com/tensorflow/tensorflow/issues/11674
    if type(features) is dict:
        feature_columns = tf.feature_column.numeric_column('inputs', [300, 3])
        input_layer = tf.feature_column.input_layer(features, feature_columns)
    else:
        input_layer = features
    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    # simple model structure for demo
    curr_layer = tf.reshape(input_layer, [-1, 900])
    curr_layer = tf.layers.dense(curr_layer, 128, 
                                 activation=tf.nn.elu, 
                                 kernel_initializer=tf.variance_scaling_initializer())
    curr_layer = tf.layers.dropout(curr_layer, 0.5, training=is_training)
    curr_layer = tf.layers.dense(curr_layer, 128, 
                                 activation=tf.nn.elu, 
                                 kernel_initializer=tf.variance_scaling_initializer())
    curr_layer = tf.layers.dropout(curr_layer, 0.5, training=is_training)
    logits = tf.layers.dense(curr_layer, N_CLASSES, activation=None, 
                                   kernel_initializer=tf.variance_scaling_initializer())
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        # tf.argmax won't work due to the incompability of tf 1.4 and 1.2
#         predicted_labels = tf.argmax(logits, axis=1, output_type=tf.int32)
        _, predicted_labels = tf.nn.top_k(logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_labels)
        eval_metric_ops = {
            "accuracy": accuracy
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "logits": logits,
            "labels": predicted_labels
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

def train():
    if tf.gfile.Exists(MODEL_DIR):
        tf.gfile.DeleteRecursively(MODEL_DIR)
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR)

    train_dataset = list(zip(np.random.rand(1000, 300, 3), np.repeat([0, 1], [500, 500])))
    train_input_fn = get_input_fn(train_dataset, repeats=2, shuffle=True, batch_size=64)
    model.train(input_fn=train_input_fn)
    return model

def export(model):
    """Export the model by opening the estimator, due to some issue 
    reported [here](https://github.com/tensorflow/tensorflow/issues/11674), 
    I find it more convinient to use the old-school way for now.
    """
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 300, 3])
    y = model_fn(tf.estimator.ModeKeys.PREDICT, x, labels=None).predictions
    logits = y['logits']
    labels = y['labels']

    model_inputs = tf.saved_model.utils.build_tensor_info(x)
    model_outputs_logits = tf.saved_model.utils.build_tensor_info(logits)
    model_outputs_labels = tf.saved_model.utils.build_tensor_info(labels)
    model_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                model_inputs
        },
        outputs={
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                model_outputs_labels,
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                model_outputs_logits
        },
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
    )


    if tf.gfile.Exists(EXPORT_DIR):
        tf.gfile.DeleteRecursively(EXPORT_DIR)
        
    builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        last_checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
        saver.restore(sess, last_checkpoint)
        builder.add_meta_graph_and_variables(
            sess, 
            tags=[tf.saved_model.tag_constants.SERVING], 
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature
            }
        )
        builder.save()

def main(_):
    model = train()
    export(model)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)