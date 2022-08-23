
ARG BASE_IMAGE=tensorflow/tensorflow:1.14.0-gpu-py3
FROM $BASE_IMAGE

# Install and expose redis default port
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
        gcc \
        ffmpeg \
        libsm6 \
        libxext6

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


# Install yolo5 requirements
COPY ./yolov5/requirements.txt /yolov5_requirements.txt
RUN pip3 install -r ./yolov5_requirements.txt

COPY ./requirements.txt /
RUN pip3 install -r ./requirements.txt

# Copy your API code
COPY ./animal_detection_api /app/animal_detection_api/
RUN true

# Copy yolo5 code
COPY ./yolov5 /app/yolov5/
RUN true

COPY ./supervisord.conf /etc/supervisord.conf
RUN true

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh


ENV PYTHONPATH="${PYTHONPATH}:/app/animal_detection_api/:/app/yolov5/"
ENV PYTHONUNBUFFERED=TRUE

# Expose the port that is to be used when calling your API

# If you change the port here, remember to update supervisord.conf 
EXPOSE 1213

# Starts up the detector, Flask app and Redis server
ENTRYPOINT [ "/startup.sh" ]
