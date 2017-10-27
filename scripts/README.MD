## Scripts
- `create_datalab_gpu.sh`: Create a gpu instance for datalab
    - need to define the zone, e.g., `gcloud config set compute/zone asia-east1-a`, where gpu instance is supported in the zone
    - it takes a very long time to setup, and it uses nvidia-docker to setup the instance
    - an alternative is to just start a gpu compute instance, with cuda and cudnn setup
- `ssh_tunnel_instance.sh`: Create a ssh tunnel between "gcloud shell" and "compute instance"
    - this is useful if you cannot get connected directly to the compute instance via a cusotmized port (e.g., 8888 for jupyter),when you are behind a firewall (e.g., the compute instance ip is not whitelisted).
    - this tunnel forward requests from cloud shell web view (8001 by default) to the customized port (8888 by default) running on compute instance
