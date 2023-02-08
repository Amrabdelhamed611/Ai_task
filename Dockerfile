FROM pytorch/pytorch:latest
WORKDIR /app

RUN mkdir Aiactive
WORKDIR Aiactive

ADD train/* .
ADD test/* .
ADD Experiments/* .

RUN conda install -c conda-forge timm tqdm flask


