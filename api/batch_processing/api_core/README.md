# Camera trap batch processing API developer readme


## Build

Navigate to the current directory `api/batch_processing/api_core`.

Modify the Docker image tag `-t` and the configuration file name `API_CONFIG` of the instance you want to build (see the `api_instances_config` folder)

```bash
export IMAGE_NAME=yasiyu.azurecr.io/camera-trap/3.0-detection-batch-internal:1

sudo docker build . --build-arg API_CONFIG=api_config_internal.py -t $IMAGE_NAME
```

If you need to debug the environment set up interactively, comment out the entry point line at the end of the Dockerfile, build the Docker image, and start it interactively:
```bash
sudo docker run -p 6000:1212 -it $IMAGE_NAME /bin/bash
```

And start the gunicorn server program manually:
```bash
gunicorn -b 0.0.0.0:1212 runserver:app
```

## Deploy

Modify the port number to expose from this server VM (set to `6000` below). The second port number is the port exposed by the Docker container, specified in [Dockerfile](Dockerfile).

Can also specify a new path for the log file to append logs to. 

```bash
sudo docker run -p 6000:1212 $IMAGE_NAME |& tee -a /home/username/foldername/batch_api_logs/log_internal_20200707.txt

```


## Testing

