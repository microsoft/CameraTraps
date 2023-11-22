# Camera trap batch processing API developer readme


## Build the Docker image for Batch node pools

We need to build a Docker image with the necessary packages (mainly TensorFlow) to run the scoring script. Azure Batch will pull this image from a private container registry, which needs to be in the same region as the Batch account. 

Navigate to the subdirectory `batch_service` (otherwise you need to specify the Docker context).

Build the image from the Dockerfile in this folder:
```commandline
export IMAGE_NAME=cameratracrsppftkje.azurecr.io/tensorflow:1.14.0-gpu-py3
export REGISTRY_NAME=cameratracrsppftkje
sudo docker image build --rm --tag $IMAGE_NAME --file ./Dockerfile .
```

Test that TensorFlow can use the GPU in an interactive Python session:
```commandline
sudo docker run --gpus all -it --rm $IMAGE_NAME /bin/bash

python
import tensorflow as tf
print('tensorflow version:', tf.__version__)
print('tf.test.is_gpu_available:', tf.test.is_gpu_available())
quit()
``` 
You can now exit/stop the container.

Log in to the Azure Container Registry for the batch API project and push the image; you may have to `az login` first:
```commandline
sudo az acr login --name $REGISTRY_NAME

sudo docker image push $IMAGE_NAME
```


## Create a Batch node pool

We create a separate node pool for each instance of the API. For example, our `internal` instance of the API has one node pool.

Follow the notebook [api_support/create_batch_pool.ipynb](../api_support/create_batch_pool.ipynb) to create one. You should only need to do this for new instances of the API.


## Flask app

The API endpoints are in a Flask web application, which needs to be run in the conda environment `cameratraps-batch-api` specified by [environment-batch-api.yml](environment-batch-api.yml). 

In addition, the API uses the `sas_blob_utils` module from the `ai4eutils` [repo](https://github.com/microsoft/ai4eutils), so that repo folder should be on the PYTHONPATH. 

Make sure to update the `API_INSTANCE_NAME`, `POOL_ID`, `BATCH_ACCOUNT_NAME`, and `BATCH_ACCOUNT_URL` values in [server_api_config.py](./server_api_config.py) to reflect which instance of the API is being deployed.

To start the Flask app in development mode, first source `start_batch_api.sh` to retrieve secrets required for the various Azure services from KeyVault and export them as environment variables in the current shell:
```commandline
source start_batch_api.sh
```

You will be prompted to authenticate via AAD (you need to have access to the AI4E engineering subscription).

Set the logs directory as needed, and the name of the Flask app:
```
export LOGS_DIR=/home/otter/camtrap/batch_api_logs
export FLASK_APP=server
```

To start the app locally in debug mode:
```commandline
export FLASK_ENV=development
flask run -p 5000 --eager-loading --no-reload
```

To start the app on a VM, with external access:
```commandline
flask run -h 0.0.0.0 -p 6011 --eager-loading --no-reload |& tee -a $LOGS_DIR/log_internal_dev_20210216.txt
```

To start the app using the production server:
```commandline
gunicorn -w 1 -b 0.0.0.0:6011 --threads 4 --access-logfile $LOGS_DIR/log_internal_dev_20210218_access.txt --log-file $LOGS_DIR/log_internal_dev_20210218_error.txt --capture-output server:app --log-level=info
```
The logs will only be written to these two log files and will not show in the console.

The API should work with more than one process/Gunicorn worker, but we have not tested it. 


## Send daily activity summary to Teams

Running [api_support/start_summarize_daily_activities.sh](../api_support/start_summarize_daily_activities.sh) will retrieve credentials from the KeyVault (you need to authenticate again) and run a script to send a summary of images processed on *all* instances of the API in the past day to a Teams webhook.
