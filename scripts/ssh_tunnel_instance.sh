#!/usr/bin/env bash
INSTANCE_NAME=$1
gcloud compute ssh $INSTANCE_NAME --ssh-flag=“-L 8081:localhost:8888”
