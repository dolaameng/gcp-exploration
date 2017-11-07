from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim import nets
from tensorflow.contrib import slim
from tensorflow.contrib import lookup
import os

from PIL import Image
from scipy.misc import imresize

tf.flags.DEFINE_string('--model_dir', './model', 'folder with model checkpoint file')
tf.flags.DEFINE_string('--export_dir', './exported_model', 'folder to export to')
tf.flags.DEFINE_string('--version', 1, 'exported model version')
FLAGS = tf.flags.FLAGS

MODEL_PATH = os.path.join(FLAGS.model_dir, 'inception_v3.ckpt')
CLASSNAME_PATH = os.path.join(FLAGS.model_dir, 'synset_words.txt')
EXPORT_DIR = os.path.join(FLAGS.export_dir, "%i/" % FLAGS.version)

IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNELS = 3

def main(_):

    # model graph
    tf.logging.info('build model graph ...')
    tf.reset_default_graph()
    tf_image = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    with slim.arg_scope(nets.inception.inception_v3_arg_scope()):
        tf_logits, tf_endpoints = nets.inception.inception_v3(tf_image, 
                                                              is_training=False, 
                                                              num_classes=1001)
    table = lookup.index_to_string_table_from_tensor(
        tf.constant(open(CLASSNAME_PATH).readlines()))
    tf_top5_logits, tf_top5_indices = tf.nn.top_k(tf_logits, k=5)
    tf_top5_indices = tf.to_int64(tf_top5_indices)
    tf_top5_class_names = table.lookup(tf_top5_indices-1) # one-based rather than zero
    saver = tf.train.Saver()
    tf.logging.info('... build model graph done')
    
    # model weights and exports
    tf.logging.info('export model ...')
    with tf.Session() as sess:
        # load weights
        sess.run(tf.tables_initializer())
        saver.restore(sess, MODEL_PATH)

        # build model signature
        model_inputs = tf.saved_model.utils.build_tensor_info(tf_image)
        model_outputs_logits = tf.saved_model.utils.build_tensor_info(tf_top5_logits)
        model_outputs_classnames = tf.saved_model.utils.build_tensor_info(tf_top5_class_names) 
        model_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    model_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    model_outputs_logits,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    model_outputs_classnames
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        )

        # save meta and variables
        if tf.gfile.Exists(EXPORT_DIR):
            tf.gfile.DeleteRecursively(EXPORT_DIR)
        builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)
        builder.add_meta_graph_and_variables(
            sess, 
            tags=[tf.saved_model.tag_constants.SERVING], 
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature
            }, 
            legacy_init_op=tf.tables_initializer()
        )
        builder.save()
    tf.logging.info('... export model done')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)