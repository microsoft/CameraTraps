# Building the Docker image for Batch node pools

We build a Docker image with the necessary packages to run the scoring script, as well as the scoring scripts themselves.

Azure Batch needs to pull the image from a private registry in the same region as the Batch account. 

The commands below are for a Linux machine with Docker installed. Navigate to this folder.

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
