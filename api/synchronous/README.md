# Camera trap real-time API


We also expose our animal detection and species classification (the latter is to come!) models through a synchronous API to support real-time use cases and our demo web app.


## Set-up

All paths are relative to this directory `api/synchronous`

(Do not need `sudo` if you added the user to the Docker group)

Prepare the model files and configuration

- Download the MegaDetector model files (the `.pb` files) to `api_core/animal_detection_classification_api/model`

- Download the classification models to TBD

- Download the class names lists to `api_core/animal_detection_classification_api/class_names`

- Modify `api_core/animal_detection_classification_api/api_config.py` to point to the desired model files.

Build the Docker image

- Clone the API Framework repo, and there, in `Containers/base-py/Dockerfile`, remove `RUN easy_install3 pip` and add `python3-pip` to the list of packages installed in the `RUN apt-get` command.

- Build our custom base Docker image to solve TensorFlow version and GPU finding issues. From the Framework repo's `Containers` directory,
```bash
sudo docker build . -f base-py/Dockerfile --build-arg BASE_IMAGE=tensorflow/tensorflow:1.14.0-gpu-py3 -t yasiyu.azurecr.io/aiforearth/tensorflow:1.14.0-gpu-py3
```

We call our base image `yasiyu.azurecr.io/aiforearth/tensorflow:1.14.0-gpu-py3` and it's used as the base image in the API's Dockerfile.

- Name the API's Docker image; modify its version and build number as needed:
```bash
export API_DOCKER_IMAGE=yasiyu.azurecr.io/camera-trap/2.0-detection-sync:1
```

- From `api_core` (Docker context is that directory), run

```bash
sudo sh build_docker.sh $API_DOCKER_IMAGE
```

- Start the Docker container to host the API locally at port 6002 of the VM:
```bash
sudo docker run --gpus all -p 6002:1212 $API_DOCKER_IMAGE
```

Now test the locally deployed API (see [Testing](#testing)). Then push the image to a container registry so we can deploy it on the production cluster.

```bash
sudo az acr login --name name_of_registry

sudo docker push $API_DOCKER_IMAGE
```


## Deployment


## Testing

From this directory (`synchronous`),

```bash
python test_synchronous_api.py "url_of_api"
```

The URL looks like `http://vm-name.eastus.cloudapp.azure.com:6002/v1/camera-trap/sync/`

Also need to provide an API key to test the API in production:

```bash
python test_synchronous_api.py "url_of_api" "api_key"
```
