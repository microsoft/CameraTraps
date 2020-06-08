FROM yasiyu.azurecr.io/aiforearth/tensorflow:1.14.0-gpu-py3

# the base image uses Python 3.6. TODO use TF2 for inference

# Install additional packages
RUN pip3 install --upgrade pip

RUN pip3 install \
    matplotlib \
    pillow \
    requests_toolbelt \
    tqdm \
    humanfriendly \
    jsonpickle
# numpy and requests are already included in the base image

# Copy your API code
COPY ./animal_detection_classification_api /app/animal_detection_classification_api/

COPY ./supervisord.conf /etc/supervisord.conf

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

# Enter your Application Insights instrumentation key to enable Azure monitoring.  You can find the key here:
#   https://docs.microsoft.com/en-us/azure/azure-monitor/app/create-new-resource#copy-the-instrumentation-key
ENV APPINSIGHTS_INSTRUMENTATIONKEY=f85104ff-1da9-4301-8f77-5b24c68788aa \
    TRACE_SAMPLING_RATE=1.0

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER="AI4E_camera_trap" \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME="camera_trap_sync" \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.7 \
    ENVSERVICE_MODEL_VERSION=1.0 \
    DISABLE_CURRENT_REQUEST_METRIC=False

# This is the prefix for your API.
# In this example, the default_post function has api_path set to '/example', so the URL will be:
#   /v1/my_api/tasker/example - API_PREFIX + api_path
# The supervisor.conf file sets the webserver port to 1212, so if you run this locally, the complete URL will be:
#   http://localhost:1212/v1/my_api/tasker/example
ENV API_PREFIX=/v1/camera-trap/sync

ENV PYTHONPATH="${PYTHONPATH}:/app/animal_detection_classification_api/"
ENV PYTHONUNBUFFERED=TRUE

# Expose the port that is to be used when calling your API
EXPOSE 1212

HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost:1212/${API_PREFIX}/ || exit 1
ENTRYPOINT [ "/startup.sh" ]
# Comment out the line above and uncomment the following line to debug and see output messages.
# ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:1212", "runserver:app"]