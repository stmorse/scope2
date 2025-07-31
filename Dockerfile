FROM docker.io/python:3.12-slim

WORKDIR app/

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt
