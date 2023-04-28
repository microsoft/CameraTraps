# Camera trap real-time flask-redis API

## Sample notebook

This README documents the configuration of the MegaDetector API; a notebook that demonstrates the *calling* of the API is available [here](camera_trap_flask_api_test.ipynb).

## Setup

### Prerequisites

The most notable prerequisite is nvidia-docker; install according to:

<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>


### Clone this repo

```bash
git clone "https://github.com/ecologize/CameraTraps/"
cd CameraTraps
```

    
### Download the model file

Download the MegaDetector model file(s) to `api/synchronous/api_core/animal_detection_api/model`.  We will download both MDv5a and MDv5b here, though currently the API is hard-coded to use MDv5a.

```bash
wget "https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt" -O api/synchronous/api_core/animal_detection_api/model/md_v5a.0.0.pt
wget "https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt" -O api/synchronous/api_core/animal_detection_api/model/md_v5b.0.0.pt
```

### Enable API key authentication (optional)

To authenticate the API via a key, create a file with name `allowed_keys.txt`, add it to the folder `api/synchronous/api_core/animal_detection_api`, then add a list of keys to the file, with one key per line.
 
 
### Build the Docker image

- Switch to the `api/synchronous/api_core` folder, from which the Docker image expects to be built.

    ```bash
    cd api/synchronous/api_core
    ```

- Name the API's Docker image (the name doesn't matter, having a name is just a convenience if you are experimenting with multiple versions, but subsequent steps will assume you have set this environment variable to something).

    ```bash
    export API_DOCKER_IMAGE=camera-trap-api:1.0
    ```

- Select the Docker base image... we recommend this one:

    ```bash
    export BASE_IMAGE=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
    ```

- If you use our recommended base image, skip this step.  If you choose a different base image that does not include PyTorch, you will need to make sure PyTorch gets installed.  The easiest way to do this is to edit api/synchronous/api_core/requirements.txt, and add the following to the end:

    ```bash
    torch==1.10.1
    torchvision==0.11.2
    ```
- Build the Docker image using build_docker.sh.

    ```bash
    sudo sh build_docker.sh $BASE_IMAGE $API_DOCKER_IMAGE
    ```

Building may take 5-10 minutes.

### Run the Docker image

The following will run the API on port 5050, but you can change that to the port on which you want to run the API.

- For GPU environments:

    ```bash
    sudo nvidia-docker run -it -p 5050:1213 $API_DOCKER_IMAGE
    ```

- For non-GPU environments:

    ```bash
    sudo docker run -it -p 5050:1213 $API_DOCKER_IMAGE
    ```

## Test the API in Postman

- To test in Postman, in a Postman tab, enter the URL of the API, e.g.:

  `http://100.100.200.200:5050/v1/camera-trap/sync/detect`
  
 - Select `POST`.
 - Optionally add the `min_confidence` parameter, which sets the minimum detection confidence that's returned to the caller (defaults to 0.1).
 - Optionally add the `min_rendering_confidence` parameter, which sets the minimum detection confidence that's rendered to returned images (defaults to 0.8) (not meaningful if "render" is False).
 - Optionally add the `render` parameter, set to `true` if you would like the images to be rendered with bounding boxes.
 - If you enabled authentication by adding the file `allowed_keys.txt` under `api/synchronous/api_core/animal_detection_api`then in the headers tab add the `key` parameter and enter the key value (this would be one of the keys that you saved to the file `allowed_keys.txt`).
 - Under `Body` select `form-data`, and create one key/value pair per image, with values of type "file" (to upload an image file).  To create a k/v pair of type "file", hover over the right side of the box where it says "key"; a drop-down will appear where you can select "file".
 - Click `Send`.

### Setting header options

![Test in postman](images/postman_url_params.jpg) 

### Specifying an API key

![Test in postman](images/postman_api_key.jpg)

### Sending one or more images

![Test in postman](images/postman_formdata_images.jpg)

<br/>

