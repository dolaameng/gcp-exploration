#!/usr/bin/env bash
INSTANCE_NAME=$1
datalab beta create-gpu --disk-size-gb 50 --machine-type n1-standard-2 --accelerator-type nvidia-tesla-k80 --accelerator-count 1 $INSTANCE_NAME

