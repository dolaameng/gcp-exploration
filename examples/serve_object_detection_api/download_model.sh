#!/usr/bin/env bash

PATH_TO_CKPT="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz"
PATH_TO_LABELS="https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"



wget -P ./model/ $PATH_TO_CKPT
wget -P ./model/ $PATH_TO_LABELS