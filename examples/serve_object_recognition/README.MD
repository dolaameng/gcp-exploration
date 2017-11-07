# Serve Object Recogntion Model on Cloud

## Step 1 Download the pretrained model
- Run `bash download_model.sh`

## Step 2 Export tf.slim model (Inception) to SavedModel Format
- Run `python export_model.py --model_dir model --export_dir exported_model --version 1`

## Step 3 Serve it on the cloud
- `export REMOTE_MODEL_DIR={your-model-dir-on-gs}`
- `gsutil cp -r ./exported_model $REMOTE_MODEL_DIR`
- `gcloud ml-engine models create object_recognition --regions us-central1`
- `gcloud ml-engine versions create v1 --model object_recognition --origin $REMOTE_MODEL_DIR/exported_model/1 --runtime-version 1.2`

## Step 4 Test the deployed model
- `export GOOGLE_PROJECT_NAME={your-project-name}`
- `export MODEL_NAME=object_recognition`
- run `python test_model_service.py`
