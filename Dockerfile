FROM python:3.8-slim  
  
ARG GRADIO_SERVER_NAME="0.0.0.0"  
ENV GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME}  
  
ARG GRADIO_SERVER_PORT="80"  
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}  
      
RUN apt-get update && \    
    apt-get install -y --no-install-recommends  \ 
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg && \
    apt-get clean && \    
    rm -rf /var/lib/apt/lists/* 
    
WORKDIR /app    
    
COPY . /app    
     
RUN rm -rf /tmp/*    
    
RUN pip install --no-cache-dir PytorchWildlife

EXPOSE 80

