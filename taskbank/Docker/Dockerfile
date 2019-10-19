FROM nvidia/cudagl:10.0-devel-ubuntu18.04
RUN apt-get update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
         build-essential \
         cmake \
         git \
         curl \
         screen \
         byobu \
         htop \
         zsh \
         vim \
         ca-certificates \
         libjpeg-dev \
         libboost-all-dev \
         libsuitesparse-dev \
         libfreeimage-dev \
         libgoogle-glog-dev \
         libgflags-dev \
         libglew-dev \
         freeglut3-dev \
         qt5-default \
         libxmu-dev \
         libxi-dev \
         libatlas-base-dev \
         libsuitesparse-dev \
         libqt5svg5 \
         libqt5svg5-dev \
         libqt5opengl5-dev \
         python3.4 \
         libpng-dev &&\
    rm -rf /var/lib/apt/lists/*
