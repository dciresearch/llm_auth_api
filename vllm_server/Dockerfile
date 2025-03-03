FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /workdir
RUN pip install -U vllm

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


COPY ./runner.sh /workdir/

ENTRYPOINT ["./runner.sh"]