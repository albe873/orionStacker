FROM rocm/dev-ubuntu-24.04:6.3
RUN apt update
RUN apt install -y build-essential libcfitsio-dev wget
RUN apt install -y cmake
RUN apt install -y libopencv-dev

# CUDA installation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN apt install -y ./cuda-keyring_1.1-1_all.deb
RUN apt update
RUN apt install -y cuda-toolkit=12.6.0-1

# Set CUDA environment variables
ENV PATH=${PATH}:/usr/local/cuda-12.6/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.6/lib64

# Install HIP ROCM SDK
RUN apt install -y rocm-hip-sdk

# User setup
#RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user
#WORKDIR /home/user