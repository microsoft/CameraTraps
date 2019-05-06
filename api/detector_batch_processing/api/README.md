# Detector batch processing API

## Deployment (internal)

Currently only people with access to our internal Azure subscription can deploy this service as the compute cluster and storage account are hosted there.

If you just need to restart the service and have access to a built Docker container, you can jump to Step 4. 

If you would like to create a new deployment of this API with its own compute resources, please create a new AML workspace and two storage containers within one storage account.


### Step 1. Fill out credentials and AML workspace info

In the `Dockerfile`, fill out the missing fields, but **remember to not commit any credentials to git!**:

- `AZUREML_PASSWORD`: this is the password for this application (which is the batch processing API) to access AML. Go to Azure Portal, Azure Active Directory, App registrations, the application representing this API (`camera-trap-async-api`) in the list, and lastly the Certificates & secrets tab. There you can add a new client secret. Keep this safely as it will not appear completely again in the portal, and set it as the `AZUREML_PASSWORD` environment variable in the `Dockerfile`.

- `STORAGE_ACCOUNT_NAME`: name of the internal Azure Storage account used for the two storage containers for storing sharded and aggregated result files.

- `STORAGE_ACCOUNT_KEY`: key to that account; cannot be a SAS key.

In `orchestrator_api/api_config.py`, fill out information on the AML workspace and storage containers:

- replace `INTERNAL_CONTAINER` with the name of the container in the storage account specified in `Dockerfile` (see `STORAGE_ACCOUNT_NAME` above) that you created for storing outputs presented to the user. Replace `AML_CONTAINER` with the name of another container (can be the same container) in that same account for storing sharded outputs of the AML jobs.

- fill out the top six fields in `AML_CONFIG` (up to and including `model_name`) with the ID of the Azure subscription used for the AML instance, region, resource group and workspace name of the AML workspace you created for this deployment, the name of the compute target in that AML workspace, and the name of the model reigstered and uploaded to the workspace. 

- also in the `AML_CONFIG` dict, fill out the `tenant-id` and `application-id`, which is near where `AZUREML_PASSWORD` is found in Azure Portal. They are called "Directory (tenant) ID" and "Application (client) ID" respectively.


### Step 2. Log in to Azure Container Registry

Authenticate to the Azure Container Registry `ai4eregistry.azurecr.io`, since in `Dockerfile` we use a base image from there. This is an older version of base image - we cannot use newer versions before updating `runserver.py` to use the decorator based AI4E API Framework. There is a DockerHub equivalent base image and we will use that instead in the next version, to be consistent with our grantees.

You can do this at the command line (where Azure CLI is installed) of the VM where you're planning to host this API:
```
az acr login --name ai4eregistry
```
You need to have the subscription where this registry is set as the default subscription.


### Step 3. Build the Docker container
Now you're all set to build the container.

Navigate to the current directory (`detector_batch_processing/api`) where the `Dockerfile` is.

```
docker build . -t name.azurecr.io/camera-trap-detection-sync:2
```

You can supply your own tag (`-t` option) and build number. You may need to use `sudo` with this command.


### Step 4. Launch the Docker container
To launch the service:

```
docker run -p 6000:80 name.azurecr.io/camera-trap-detection-batch:2 |& tee -a camera-trap-api-async-log/log20190415.txt
```

Substitute the tag of the image you built in the last step (or that of a pre-built one), the port you'd like to expose the API at (6000 above), and specify the location to store the log messages (printed to console too).

You may need to use `sudo` with this command.


## Work items

- [ ] Rename `aml_config_scripts` to `aml_scripts` now that the cluster config file is no longer used

- [ ] Make use of Key Vault to access crendentials
