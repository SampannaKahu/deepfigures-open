#!/bin/bash -ex

sudo sed -i -e 's/^Defaults\tsecure_path.*$//' /etc/sudoers

# Check Python
sudo apt-get install software-properties-common python-software-properties -y
sudo add-apt-repository ppa:jonathonf/python-3.6 -y
sudo apt-get update
sudo apt-get install python3.6 -y
sudo apt-get install python3-setuptools -y
sudo apt install python3-pip -y

echo "Python Version:"
python3 --version
pip3 install sregistry
sregistry version

echo "sregistry Version:"

# Install Singularity

SINGULARITY_BASE="${GOPATH}/src/github.com/sylabs/singularity"
export PATH="${GOPATH}/bin:${PATH}"

mkdir -p "${GOPATH}/src/github.com/sylabs"
cd "${GOPATH}/src/github.com/sylabs"

git clone -b vault/release-3.3 https://github.com/sylabs/singularity
cd singularity
./mconfig -v -p /usr/local
make -j `nproc 2>/dev/null || echo 1` -C ./builddir all
sudo make -C ./builddir install

mkdir -p ~/.singularity
echo $SYLABS_API_TOKEN >> ~/.singularity/sylabs-token