FROM tensorflow/tensorflow@sha256:37c7db66cc96481ac1ec43af2856ef65d3e664fd7f5df6b5e54855149f7f8594
# The above is the SHA digest for tensorflow/tensorflow:latest-gpu.
# At the time of writing this, the latest tensorflow version was 2.3.1

ENV LC_ALL C.UTF-8
ENV IN_DOCKER Yes
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get update
RUN apt-get install -y libjasper1 libjasper-dev

# install system packages

RUN apt-get clean \
 && apt-get update --fix-missing \
 && apt-get install -y \
    git \
    curl \
    gcc \
    build-essential \
    tcl \
    g++ \
    zlib1g-dev \
    libjpeg8-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng12-dev \
    tcl-dev \
    tk-dev \
    python3 \
    python3-pip \
    python3-tk \
    ghostscript \
    openjdk-8-jre \
    poppler-utils \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    texlive-font-utils \
    vim \
 && rm -rf /var/lib/apt/lists/*

