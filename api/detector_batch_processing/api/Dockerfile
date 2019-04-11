FROM ai4eregistry.azurecr.io/1.0-base-py-ubuntu16.04:latest

RUN echo "source activate ai4e_py_api" >> ~/.bashrc \
    && conda install -c conda-forge -n ai4e_py_api numpy pandas

# Azure blob packages should already be installed in the base image. Just need to install Azure ML SDK
RUN pip install --upgrade azureml-sdk

# Note: supervisor.conf reflects the location and name of your api code.
# If the default (./my_api/runserver.py) is renamed, you must change supervisor.conf
COPY ./supervisord.conf /etc/supervisord.conf
# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

COPY ./LocalForwarder.config /lf/

# Copy your API code
COPY ./orchestrator_api /app/orchestrator_api/

# Add this API directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/"

# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    APPINSIGHTS_LIVEMETRICSSTREAMAUTHENTICATIONAPIKEY= \
    LOCALAPPDATA=/app_insights_data \
    OCAGENT_TRACE_EXPORTER_ENDPOINT=localhost:55678

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER=AI4E_Test \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=batch_animal_detection \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    SERVICE_MODEL_VERSION=1.0

ENV API_PREFIX=/v1/camera-trap/detection-batch

ENV AZUREML_PASSWORD=

ENV STORAGE_ACCOUNT_NAME= \
    STORAGE_ACCOUNT_KEY=

# Expose the port that is to be used when calling your API
EXPOSE 80
#HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
#  CMD curl -f http://localhost/ || exit 1
ENTRYPOINT [ "/startup.sh" ]
