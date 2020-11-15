FROM sampyash/deepfigures:dependencies_gpu_0.0.1

WORKDIR /work

# Set up deepfigures weights (source: https://github.com/allenai/deepfigures-open).
RUN mv /weights /work

# Set up AWS credentials for data generation.
RUN mkdir ~/.aws
ADD credentials /work
RUN cat /work/credentials > ~/.aws/credentials

# Add the source code.
ADD . /work

# Setup stitch_rects
RUN cd vendor/tensorboxresnet/tensorboxresnet/utils && make && cd ../../../..

CMD [ "/bin/bash" ]
