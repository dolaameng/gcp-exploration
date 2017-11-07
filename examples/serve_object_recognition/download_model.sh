#!/usr/bin/env bash

MODEL_URL="http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"

rm -fr ./model
mkdir model

wget -P ./model/ $MODEL_URL
tar -xzf model/inception_v3_2016_08_28.tar.gz -C ./model/
wget -P ./model/ https://raw.githubusercontent.com/dolaameng/data/master/synset_words.txt