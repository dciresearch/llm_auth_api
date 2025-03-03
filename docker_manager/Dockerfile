FROM python:3.10

RUN apt-get update
RUN apt install docker.io -y
RUN docker --version

RUN pip install fastapi
RUN pip install docker
RUN pip install uvicorn
RUN pip install fire
RUN pip install sqlalchemy
RUN pip install pyyaml
RUN pip install celery
RUN pip install openai
RUN pip install redis
RUN pip install sqlite-web

WORKDIR /workdir