
ARG BASE_IMAGE=tensorflow/tensorflow:1.14.0-gpu-py3
FROM $BASE_IMAGE

# install and expose redis default port
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
        apt-transport-https \
        build-essential \
        gcc
# the base image uses Python 3.6. TODO use TF2 for inference

# Install additional packages
RUN pip3 install --upgrade pip

RUN pip3 install \
    redis \
    numpy \
    requests \
    matplotlib \
    pillow \
    requests_toolbelt \
    tqdm \
    humanfriendly\
    jsonpickle


COPY ./requirements.txt /
RUN pip3 install -r ./requirements.txt

# Copy your API code
COPY ./animal_detection_api /app/animal_detection_api/
RUN true

COPY ./supervisord.conf /etc/supervisord.conf
RUN true

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh


ENV PYTHONPATH="${PYTHONPATH}:/app/animal_detection_api/"
ENV PYTHONUNBUFFERED=TRUE

# Expose the port that is to be used when calling your API

#!!! IF YOU CHANGE THE PORT HERE REMEMBER TO UPDATE supervisord.conf FILE AS WELL
EXPOSE 1212

# starts the supervisor.conf (which starts up the detector, flask app and redis server)
ENTRYPOINT [ "/startup.sh" ]
