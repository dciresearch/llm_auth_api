# FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
# FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
# FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y python3
RUN apt-get -y install python3-pip

WORKDIR /workdir
# RUN pip install vllm==0.11.0
#RUN pip install vllm==0.8.5.post1
# RUN pip install vllm==0.14.0
RUN pip install vllm==0.19.1

# TODO replace with ==0.17.0
# RUN pip install vllm --extra-index-url https://wheels.vllm.ai/nightly

RUN pip install torch-c-dlpack-ext



# ==0.7.3
# RUN pip install nvidia-nccl-cu12==2.26.5
# RUN pip install nvidia-nccl-cu12==2.28.7

RUN apt-get install python-is-python3

COPY ./runner.sh /workdir/

ENTRYPOINT ["./runner.sh"]