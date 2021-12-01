# Camera trap real-time flask-redis API

## Set-up

Prepare the model files

- Download the MegaDetector model files (the `.pb` files) to `api_flask_redis/api_core/animal_detection_api/model`

Build the Docker image

- Clone the API Framework repo, navigate to  folder `api_flask_redis/api_core`
- Name the API's Docker image; modify its version and build number as needed:
```bash
export API_DOCKER_IMAGE=camera-trap-api:1
```

- From `api_core` (Docker context is that directory), run

```bash
sudo  docker build . -t $API_DOCKER_IMAGE
```

- Start the Docker container to host the API locally at port 5050 of the VM:
```bash
sudo nvidia-docker run -p 5050:1212 $API_DOCKER_IMAGE
```

Test the API in postman

![Test in postman](images/postman_1.jpg)

![Test in postman](images/postman_2.jpg)

