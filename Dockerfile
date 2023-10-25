FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ARG TORCH_CUDA_ARCH_LIST="7.0+PTX"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0"

# Update package list and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libyaml-cpp-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install GCC, G++ 9
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-9 \
    g++-9 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Install conda 4.12.0
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh \
    && chmod +x miniconda.sh \
    && ./miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

# Set some environment variables
ENV PATH /opt/conda/bin:$PATH
# ENV WANDB_API_KEY=...
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1

# Add OneFormer
ADD . /OneFormer
WORKDIR /OneFormer

# Install dependencies
RUN conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN pip3 install -U opencv-python
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN pip3 install git+https://github.com/cocodataset/panopticapi.git
RUN pip3 install git+https://github.com/mcordts/cityscapesScripts.git
RUN pip3 install -r requirements.txt

# Don't need wandb for inference
# RUN pip3 install wandb
# RUN wandb login

RUN pip3 install mlflow boto3 Pillow==9.5.0
# install oneformer
RUN git config --global --add safe.directory '*' && \
    pip install -e .

# Setup MSDeformAttn
RUN cd oneformer/modeling/pixel_decoder/ops && \
    sh ./make.sh

# Set the default command to run when starting the container
CMD ["/bin/bash"]
