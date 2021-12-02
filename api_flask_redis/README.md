# Camera trap real-time flask-redis API

## Setup

### Prerequisites

The most notable prerequisite is nvidia-docker; install according to:

<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>


### Set up the repo

- Clone the camera traps repo:

```bash
git clone "https://github.com/microsoft/CameraTraps/"
cd CameraTraps
```

- During this testing phase, switch to the api-flask-redis-v1 branch:

```bash
git checkout api-flask-redis-v1
````


### Prepare the model files

Download the MegaDetector model file to `api_flask_redis/api_core/animal_detection_api/model`:

```bash
wget "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb" -O api_flask_redis/api_core/animal_detection_api/model/md_v4.1.0.pb
```

### Enable API Key authentication (optional)

 - To authenticate the API via a key, add a list of keys to file (e.g GUID/UUID) `allowed_keys.txt`, one key per line
 - Then in the `config.py` file under `api_flask_redis/api_core/animal_detection_api` set `CHECK_API_KEY` to `true`

```
CHECK_API_KEY=True
```

### Build the Docker image

- Switch to the `api_flask_redis/api_core` folder, from which the Docker image expects to be built:

```bash
cd api_flask_redis/api_core
```

- Name the API's Docker image; modify its version and build number as needed:
```bash
export API_DOCKER_IMAGE=camera-trap-api:1.0
```

- set the base tensor flow image
#### For non-GPU environments

- Name the base tensorflow image
```bash
export BASE_IMAGE=tensorflow/tensorflow:1.14.0-py3
```

#### For GPU environments

- Name the base tensorflow image
```bash
export BASE_IMAGE=tensorflow/tensorflow:1.14.0-gpu-py3
```

- Build the Docker image:

```bash
sudo sh build_docker.sh $BASE_IMAGE $API_DOCKER_IMAGE
```

### Run the Docker image

#### For GPU environments

- Start the Docker container to host the API locally at port 5050 of the VM:

```bash
sudo nvidia-docker run -p 5050:1212 $API_DOCKER_IMAGE
```

#### For non-GPU environments

- Start the Docker container to host the API locally at port 5050 of the VM:

```bash
sudo docker run -p 5050:1212 $API_DOCKER_IMAGE
```

## Test the API in Postman

- To test in Postman, in a Postman tab enter the URL of the API, e.g.:

  `http://52.168.83.103:5050/v1/camera-trap/sync/detect'
  
  ?confidence=0.8&render=false`

 - Select `POST`
 - Add the `confidence` parameter, and provide a value for confidence level
 - Optionally add the `render` parameter, set to `true` if you would like the images to be rendered with bounding boxes
 - If in the config file `CHECK_API_KEY` is set to `true` then in the header's add parameter `key` and enter key value
 - Under `Body` select `form-data`, create one key/value pair per image, with values of type "file" (to upload an image file)
 - Click `Send`

![Test in postman](images/postman_url_params.jpg)

![Test in postman](images/postman_api_key.jpg)

![Test in postman](images/postman_formdata_images.jpg)
