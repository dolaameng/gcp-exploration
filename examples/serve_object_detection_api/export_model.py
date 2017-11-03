import tensorflow as tf
import os
import re

tf.flags.DEFINE_string('--model_dir', './model', 'folder containing pretrained model ssd_mobilenet_v1_coco_11_06_2017')
tf.flags.DEFINE_string('--export_dir', './exported_model', 'folder for model to be exported')
tf.flags.DEFINE_integer('--version', 1, 'exported model version')
FLAGS = tf.flags.FLAGS

MODEL_PATH = os.path.join(FLAGS.model_dir, 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb')
LABEL_PATH = os.path.join(FLAGS.model_dir, 'mscoco_label_map.pbtxt')
EXPORT_PATH = os.path.join(FLAGS.export_dir, '{0!s}/'.format(FLAGS.version))

def load_model_graph(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        model_file = tf.gfile.GFile(model_path, 'rb')
        graph_def.ParseFromString(model_file.read())
        tf.import_graph_def(graph_def, name='object_detection_api')
    return graph

def load_mscoco_labels(label_path):
    """Take a shortcut by using regular expression"""
    id2label = {}
    with open(label_path, 'r') as f:
        items = re.findall(r'item\s*\{[^}]*\}', f.read(), re.MULTILINE)
        for item in items:
            id = int(re.findall(r'id: (\d+)', item)[0])
            label = re.findall(r'display_name: "([\w\s]+)"', item)[0]
            id2label[id] = label
    return id2label

def export_model(graph, export_path):
    if tf.gfile.Exists(export_path):
        tf.gfile.DeleteRecursively(export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    with tf.Session(graph=graph) as sess:
        images = graph.get_tensor_by_name('object_detection_api/image_tensor:0')
        # images = tf.reshape(images, [-1, 100, 100, 3])
        boxes = graph.get_tensor_by_name('object_detection_api/detection_boxes:0')
        boxes = tf.reshape(boxes, [-1, -1, -1])
        scores = graph.get_tensor_by_name('object_detection_api/detection_scores:0')
        scores = tf.reshape(scores, [-1, -1])
        classes = graph.get_tensor_by_name('object_detection_api/detection_classes:0')
        classes = tf.reshape(classes, [-1, -1])

        model_inputs = tf.saved_model.utils.build_tensor_info(images)
        model_outputs_boxes = tf.saved_model.utils.build_tensor_info(boxes)
        model_outputs_scores = tf.saved_model.utils.build_tensor_info(scores)
        model_outputs_classes = tf.saved_model.utils.build_tensor_info(classes)

        model_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS: 
                    model_inputs,
            },
            outputs={
                'boxes': model_outputs_boxes,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: 
                    model_outputs_scores,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: 
                    model_outputs_classes
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess, 
            tags=[tf.saved_model.tag_constants.SERVING], 
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature
            }
        )
        builder.save()


def main(argv=None):
    tf.logging.info("loading pretrained model...")
    graph = load_model_graph(MODEL_PATH)
    id2label = load_mscoco_labels(LABEL_PATH)
    tf.logging.info("...loading pretrained model done")

    tf.logging.info('exporting model...')
    export_model(graph, EXPORT_PATH)
    tf.logging.info('...exporting model done')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)