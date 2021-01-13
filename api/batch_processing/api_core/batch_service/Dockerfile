FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install --upgrade pip
RUN pip install azure-storage-blob==12.3.0 pillow numpy

# Copy scoring script
RUN mkdir /app
COPY ./score.py /app/score.py
