FROM python:3

WORKDIR /usr/local/torch

COPY . /usr/local/torch

RUN pip install -r requirments.txt
