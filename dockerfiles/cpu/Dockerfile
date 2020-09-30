# image for running DeepFigures

FROM tensorflow/tensorflow:1.14.0-py3

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

WORKDIR /work

ADD ./requirements_cpu.txt /work

RUN pip3 install --upgrade pip \
 && pip install Cython==0.25.2
RUN pip3 install -r ./requirements_cpu.txt -f https://download.pytorch.org/whl/torch_stable.html

ADD ./vendor/tensorboxresnet /work/vendor/tensorboxresnet
#RUN pip3 install -e /work/vendor/tensorboxresnet

RUN mkdir ~/.aws
ADD credentials /work
RUN cat /work/credentials > ~/.aws/credentials

RUN pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely
RUN pip install imgaug

ADD . /work
RUN pip3 install --quiet -e /work

#ADD https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/deepfigures/weights.tar.gz /work
#RUN tar -xf /work/weights.tar.gz
#RUN rm weights.tar.gz

CMD [ "/bin/bash" ]
