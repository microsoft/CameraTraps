# Detector batch processing API

## Deployment (internal)

Currently only people with access to our internal Azure subscription can deploy this service as the compute cluster and storage account are hosted there.

You can jump to Step 4 if you have access to a built Docker container in a Docker registry.


### Step 1. Fill out credentials

Fill out the missing fields in `Dockerfile` and `orchestrator_api/api_config.py`, but **remember to not commit any credentials to git!**

In the `Dockerfile`:

- `AZUREML_PASSWORD`: this is the password for this application (which is the batch processing API) to access AML. Go to Azure Portal, Azure Active Directory, App registrations, the application representing this API (`camera-trap-async-api`) in the list, and lastly the Certificates & secrets tab. There you can add a new client secret. Keep this safely as it will not appear completely again in the portal, and set it as the `AZUREML_PASSWORD` environment variable in the `Dockerfile`.

- `STORAGE_ACCOUNT_NAME`: name of the internal Azure Storage account used to host result files both sharded and aggregated.

- `STORAGE_ACCOUNT_KEY`: key to that account; cannot be a SAS key.

In `orchestrator_api/api_config.py`:

- in the `AML_CONFIG` dict, fill out the `tenant-id` and `application-id`, which is near where `AZUREML_PASSWORD` is found in Azure Portal. They are called "Directory (tenant) ID" and "Application (client) ID" respectively.


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

```
docker run -p 6000:80 name.azurecr.io/camera-trap-detection-batch:2 |& tee -a camera-trap-api-async-log/log20190415.txt
```

Please substitute the tag of the image you built in the last step (or that of a pre-built one), the port you'd like to expose the API at (6000 above), and specify the location to store the log messages (printed to console too).

You may need to use `sudo` with this command.


## Deployment in your own Azure subscription

If you were to deploy this API in your own subscription, you would need to create an AML workspace, create a managed compute target (cluster of VMs) in that workspace, reigster and upload the detector model to that workspace, create a storage account, and create a service principle so that this API can authenticate to use the AML workspace. See the `AML_CONFIG` dict in `orchestrator_api/api_config.py`.


## Work items

- [ ] Rename `aml_config_scripts` to `aml_scripts` now that the cluster config file is no longer used

- [ ] Make use of Key Vault to access crendentials
