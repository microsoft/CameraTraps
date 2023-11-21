# Default to an image with PyTorch installed, for GPU installations.
#
# The README recommends a different image for CPU-only installations.
ARG BASE_IMAGE=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM $BASE_IMAGE

# Install Redis and expose the default Redis port 
RUN  apt-get update && apt-get install -y redis-server
EXPOSE  6379

RUN apt-get update --fix-missing \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y apt-utils \
        supervisor \
        curl \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        python3-setuptools \
        python3.7-dev \
        python3.7 \
        python3-pip \
        apt-transport-https \
        build-essential \
        gcc \
        ffmpeg \
        libsm6 \
        libxext6

# Prepare to install Python dependencies
RUN pip3 install --upgrade pip

# These are the standard MDv5 dependencies, other than PyTorch.
#
# PyTorch, CUDA, and CuDNN come from the base image.
RUN pip3 install \
    redis \
    numpy \
    requests \
    matplotlib \
    requests_toolbelt \
    pillow \
    tqdm \
    humanfriendly \
    jsonpickle \
    opencv-python \
    pandas \
    seaborn==0.11.0 \
    PyYAML==6.0 

# Commenting this out, but leaving it here for posterity.  We don't
# want to use YOLOv5's requirements file, because it will choose a
# version of PyTorch that - ironically - won't work with YOLOv5 (at
# least as of the time this Dockerfile was last edited).
#
# COPY ./yolov5/requirements.txt /yolov5_requirements.txt
# RUN pip3 install -r ./yolov5_requirements.txt

WORKDIR /

# This requirements file contains that are specific to the API, i.e.,
# not used by other parts of the repo, e.g. Flask.
COPY ./requirements.txt /
RUN true
RUN pip3 install -r /requirements.txt

# Copy API code into the container
COPY ./animal_detection_api /app/animal_detection_api/
RUN true

# Copy YOLOv5 code into the container
COPY ./yolov5 /app/yolov5/
RUN true

# Copy startup scripts/config into the container
COPY ./supervisord.conf /etc/supervisord.conf
RUN true
COPY ./startup.sh /
RUN chmod +x /startup.sh

ENV PYTHONPATH="${PYTHONPATH}:/app/animal_detection_api/:/app/yolov5/"
ENV PYTHONUNBUFFERED=TRUE

# Expose the API port
#
# If you change the port here, remember to update supervisord.conf 
EXPOSE 1213

# Starts up the detector, Flask app and Redis server
ENTRYPOINT [ "/startup.sh" ]

