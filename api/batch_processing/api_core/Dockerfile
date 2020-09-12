FROM mcr.microsoft.com/aiforearth/base-py:1.13
ARG API_CONFIG

# Install api-required packages.
RUN pip3 install \
    azure-storage-blob==2.1.0 \
    azureml-sdk==1.0.33

# Note: supervisor.conf reflects the location and name of your api code.
# If the default (./my_api/runserver.py) is renamed, you must change supervisor.conf
COPY ./supervisord.conf /etc/supervisord.conf

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

# Copy your API code
COPY ./orchestrator_api /app/orchestrator_api/

# Copy utility libraries to the AML scripts folder
COPY ./orchestrator_api/sas_blob_utils.py /app/orchestrator_api/aml_scripts/sas_blob_utils.py

# Copy the config file of the desired instance
RUN echo using the configuration file $API_CONFIG
COPY ./api_instances_config/$API_CONFIG /app/orchestrator_api/api_config.py

# Enter your Application Insights instrumentation key to enable Azure monitoring.  You can find the key here:
#   https://docs.microsoft.com/en-us/azure/azure-monitor/app/create-new-resource#copy-the-instrumentation-key
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    TRACE_SAMPLING_RATE=1.0

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER=AI4E_CameraTrapBatchAPI \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=batch_animal_detection \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.7 \
    ENVSERVICE_MODEL_VERSION=1.0 \
    DISABLE_CURRENT_REQUEST_METRIC=False

# This is the prefix for your API.
# In this example, the default_post function has api_path set to '/example', so the URL will be:
#   /v1/my_api/tasker/example - API_PREFIX + api_path
# The supervisor.conf file sets the webserver port to 1212, so if you run this locally, the complete URL will be:
#   http://localhost:1212/v1/my_api/tasker/example
ENV API_PREFIX=/v3/camera-trap/detection-batch

ENV AZUREML_PASSWORD=

ENV STORAGE_ACCOUNT_NAME= \
    STORAGE_ACCOUNT_KEY=

# Add this API directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/orchestrator_api/"
ENV PYTHONUNBUFFERED=TRUE

# Expose the port that is to be used when calling your API
EXPOSE 1212

HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost:1212/${API_PREFIX}/ || exit 1

# startup.sh uses supervisord.conf and have 4 workers
ENTRYPOINT [ "/startup.sh" ]

# Comment out the line above and uncomment the following line to debug and see output messages.
# ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:1212", "runserver:app"]

# CMD ["gunicorn", "-b", "0.0.0.0:1212", "runserver:app"]
