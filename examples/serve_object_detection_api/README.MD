# Serve Google's Object Detection API on the Cloud

## Step 1 Download pretrained object detection model
- Run `bash download_model.sh`

## Step 2 Export the pretrained model into SavedModel format
My code of exporting the model doesn't work due to some tensor shape related errors. See [here](https://github.com/tensorflow/models/issues/1811) for more details.

However, the [official repo](https://github.com/tensorflow/models/tree/master/research/object_detection) of object detection model has a script to export the model for deployment.

- Run `python export_model.py --model_dir model --export_dir exported_model --version 1`

## Step 3 Serve it on the cloud
- `export REMOTE_MODEL_DIR={your-model-dir-on-gs}`
- `gsutil cp -r ./exported_model $REMOTE_MODEL_DIR`
- `gcloud ml-engine models create object_detection --regions us-central1`
- `gcloud ml-engine versions create v1 --model object_detection --origin $REMOTE_MODEL_DIR/exported_model/1 --runtime-version 1.2`

## Step 4 Test the deployed model
- `export GOOGLE_PROJECT_NAME={your-project-name}`
- `export MODEL_NAME=object_detection`
- run `python test_model_service.py`

## Step 5 Simple Client to test it