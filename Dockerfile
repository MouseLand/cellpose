# Dockerfile by Sebastián Sterling
FROM accetto/ubuntu-vnc-xfce-g3:latest

USER root

WORKDIR /Cellpose


# Install Packages
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    vim \
    git \
    python3-pip \
    mesa-common-dev \
    libgl1-mesa-dri \
    libxcb-cursor0 \
    mesa-utils \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libturbojpeg \
    libjpeg-dev \
    zlib1g-dev \
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt upgrade -y
COPY requirements.txt .


#### Install Cellpose ####
RUN pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu126 
RUN pip3 install --break-system-packages git+https://www.github.com/mouseland/cellpose.git 
RUN pip3 install --break-system-packages 'cellpose[gui]' 
RUN pip3 install --break-system-packages cellpose
#### Install Cellpose ####


ENV VNC_PW=@Pass-Word4321 \
    VNC_RESOLUTION=1280x800

CMD ["python3", "-m", "cellpose"]

# sudo docker build -t cellpose-web .
# sudo docker run -d --name cellpose-vnc -p 36901:6901 --hostname quick cellpose-web

# To run it with GPUS
# sudo docker run -d --name cellpose-vnc --gpus all -p 36901:6901 --hostname quick cellpose-web

