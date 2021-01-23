# Camera trap batch processing API developer readme



## Building the Docker image for Batch node pools

We need to build a Docker image with the necessary packages (mainly TensorFlow) to run the scoring script. Azure Batch will pull this image from a private registry, which needs to be in the same region as the Batch account. 

Navigate to the subdirectory `batch_service` (otherwise you need to specify the Docker context).

Build the image from the Dockerfile in this folder:
```commandline
export IMAGE_NAME=cameratracrsppftkje.azurecr.io/tensorflow:1.14.0-gpu-py3
export REGISTRY_NAME=cameratracrsppftkje
sudo docker image build --rm --tag $IMAGE_NAME --file ./Dockerfile .
```

Test the scoring file (you should see the TensorFlow version and GPU availability printed out before the script errors out):
```commandline
sudo docker run --gpus all -it --rm $IMAGE_NAME /bin/bash

python /app/score.py 
``` 
You can now exit/stop the container.

Log in to the Azure Container Registry for the batch API project and push the image; you may have to `az login` first:
```commandline
sudo az acr login --name $REGISTRY_NAME

sudo docker image push $IMAGE_NAME
```

## Create a Batch node pool

We create a separate node pool for each instance of the API. For example, our `internal` instance of the API has one node pool.

Follow the notebook `create_batch_node_pool.ipynb` to create one. You should only need to do this for new instances of the API or if the node pool needs to be re-started completely.


## Flask app

The API endpoints are in a Flask web application, which needs to be run in a conda environment specified by [environment-api.yml](environment-api.yml).


