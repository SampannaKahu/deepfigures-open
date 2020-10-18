FROM sampyash/deepfigures:pdflatex_tf_gpu_2.3.1_py_3.6

RUN pip3 install --upgrade pip \
 && pip install Cython==0.25.2
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install imgaug==0.4.0
RUN pip install scipy==1.1.0
RUN pip install arrow==0.17.0
RUN pip install traitlets==4.3.3
RUN pip install tqdm==4.50.2
RUN pip install lxml==4.6.1
RUN pip install bs4==0.0.1
RUN pip install tf-slim==1.1.0
RUN pip install awscli==1.18.69
RUN pip install boto3==1.13.19

# Set up deepfigures weights (source: https://github.com/allenai/deepfigures-open).
ADD https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/deepfigures/weights.tar.gz /
RUN tar -xf /weights.tar.gz && rm /weights.tar.gz && chmod 777 weights
