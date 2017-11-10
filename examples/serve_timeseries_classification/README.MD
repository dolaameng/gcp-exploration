# Serve a customized time-series classification model via google ml-engine
- Development Env: tensorflow 1.4
- Deployment Env: ml-engine runtime 1.2 (tensorflow 1.2)

## Step 1 Build Model and Export via `tensorflow.estimator` API
- `python train_and_export.py --model_dir model --export_dir exported_model --version 1`
## Step 2 Serve it on cloud
- `export REMOTE_MODEL_DIR={your-model-dir-on-gs}`
- `gsutil cp -r ./exported_model $REMOTE_MODEL_DIR`
- `gcloud ml-engine models create timeseries --regions us-central1`
- `gcloud ml-engine versions create v1 --model timeseries --origin $REMOTE_MODEL_DIR/exported_model/1 --runtime-version 1.2`
## Step 3 Test deployment
- `export GOOGLE_PROJECT_NAME={your-project-name}`
- `export MODEL_NAME=timeseries`
- `run python test_model_service.py`
