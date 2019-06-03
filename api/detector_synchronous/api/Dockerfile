### Docker container setup

# Pull in the AI for Earth Base Image, so we can extract necessary libraries.
FROM mcr.microsoft.com/aiforearth/base-py:latest as ai4e_base

# Use any compatible Ubuntu-based image as your selected base image.
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
# Copy the AI4E tools and libraries to our container.
COPY --from=ai4e_base /ai4e_api_tools /ai4e_api_tools
ENV PATH /usr/local/envs/ai4e_py_api/bin:$PATH
ENV PYTHONPATH="${PYTHONPATH}:/ai4e_api_tools"

# Install Miniconda, Flask, Supervisor, uwsgi
RUN ./ai4e_api_tools/requirements/install-api-hosting-reqs.sh

# Install Opencensus
RUN ./ai4e_api_tools/requirements/install-opencensus.sh

# Install Application Insights
RUN ./ai4e_api_tools/requirements/install-appinsights.sh


### Package installation
RUN echo "source activate ai4e_py_api" >> ~/.bashrc \
    && conda install -c conda-forge -n ai4e_py_api numpy pandas

RUN /usr/local/envs/ai4e_py_api/bin/pip install --upgrade pip
RUN /usr/local/envs/ai4e_py_api/bin/pip install tensorflow-gpu==1.9 pillow requests_toolbelt
RUN /usr/local/envs/ai4e_py_api/bin/pip install gunicorn

### Copy the necessary files for running the API

# Note: supervisor.conf reflects the location and name of your api code.
COPY ./supervisord.conf /etc/supervisord.conf

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

COPY ./LocalForwarder.config /lf/

# Copy your API code
COPY ./animal_detection_api /app/animal_detection_api/

# Add this API directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/"


### Environment variables, App Insights and health check
# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    APPINSIGHTS_LIVEMETRICSSTREAMAUTHENTICATIONAPIKEY= \
    LOCALAPPDATA=/app_insights_data \
    OCAGENT_TRACE_EXPORTER_ENDPOINT=localhost:55678

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER="AI4E_camera_trap" \
    SERVICE_CLUSTER="Local Docker" \
    SERVICE_MODEL_NAME="camera-trap-detection-sync" \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    SERVICE_MODEL_VERSION=0.1.0

ENV API_PREFIX=/v1/camera-trap/detection-sync

# Expose the port that is to be used when calling your API
EXPOSE 8024
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost/ || exit 1

ENTRYPOINT [ "/startup.sh" ]

# Enable for guicorn testing via stdout
#ENV PYTHONUNBUFFERED=TRUE
#CMD ["gunicorn", "-b", "0.0.0.0:8024", "runserver:app"]