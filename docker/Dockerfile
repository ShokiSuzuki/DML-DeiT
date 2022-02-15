FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

ARG USER=user

RUN apt update && apt -y upgrade
RUN apt install -y --no-install-recommends \
                   python3 \
                   python3-pip \
                   vim \
                   tmux \
                   wget \
                   git \
                   curl \
                   libgl1-mesa-dev

ENV PATH=/Conda/bin:${PATH}

RUN curl --silent -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /Conda && \
    rm -f Anaconda3-2021.05-Linux-x86_64.sh
RUN echo "source /Conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN conda update --all
RUN pip3 install --upgrade pip
# RUN conda install -c pytorch pytorch torchvision
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch -c nvidia
RUN pip3 install timm==0.3.2 \
                 jupyterlab \
                 matplotlib \
                 opencv-python \
                 tensorboard \
                 umap-learn \
                 pytorch-gradcam


WORKDIR /home/${USER}

EXPOSE 8889
