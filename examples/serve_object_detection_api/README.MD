# Serve Google's Object Detection API on the Cloud

## Step 1 Download pretrained object detection model
- Run `download_model.sh`
## Step 2 Export the pretrained model into SavedModel format
- Run `python export_model.py --model_dir models --export_dir exported_model`
## Step 3 Serve it on the cloud
## Step 4 Simple Client to test it