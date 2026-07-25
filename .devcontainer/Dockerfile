FROM rocm/dev-ubuntu-24.04:7.2
RUN apt update
RUN apt install -y build-essential libcfitsio-dev wget
RUN apt install -y cmake
RUN apt install -y libopencv-dev

# CUDA installation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN sudo dpkg -i cuda-keyring_1.1-1_all.deb
RUN sudo apt-get update
RUN sudo apt-get -y install cuda-toolkit-12-9

# Set CUDA environment variables
ENV PATH=${PATH}:/usr/local/cuda-12.9/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.9/lib64

# Install HIP ROCM SDK
RUN apt install -y rocm-hip-sdk

# Install gtkmm
RUN apt install -y libgtkmm-4.0-dev

# Dependencies for building OpenCV statically from source
RUN apt install -y \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    git

# User setup
#RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user
#WORKDIR /home/user