#!/bin/bash

#install golang
#go to home directory
cd

#download go binary
wget https://dl.google.com/go/go1.10.3.linux-amd64.tar.gz

#unzip and remove
sudo tar -C /usr/local -xzf go1.10.3.linux-amd64.tar.gz
rm go1.10.3.linux-amd64.tar.gz

#export path
export PATH=$PATH:/usr/local/go/bin
echo "" >> .profile
echo "#export go path" >> .profile
echo "export PATH=$PATH:/usr/local/go/bin" >> .profile

#make root directory and set GOPATH
sudo rm -rf gopath
mkdir -p gopath/src
export GOPATH=$HOME/gopath
echo "" >> .profile
echo "#set GOPATH" >> .profile
echo "export GOPATH=$HOME/gopath" >> .profile

#export workspace bin
export PATH=$PATH:$GOPATH/bin
echo "" >> .profile
echo "#set local bin" >> .profile
echo "export PATH=$PATH:$GOPATH/bin" >> .profile
#Go should be installed now

# Pip doesn't work without this
echo "export LC_ALL=C" >> .profile

# #clone the TorMentor Repository
echo "Installing Biscotti"
cd gopath/src
git clone https://github.com/DistributedML/Biscotti.git

source ~/.profile

#Install dependencies
echo "Installing Dependencies"
go get github.com/DistributedClocks/GoVector/govec
go get github.com/kniren/gota/dataframe
go get github.com/sbinet/go-python
go get gonum.org/v1/gonum/mat
go get github.com/coniks-sys/coniks-go/crypto/vrf

# Kyber and requirements
go get github.com/dedis/kyber
cd github.com/dedis/kyber
go get -t ./... # install 3rd-party dependencies

#Installing python libraries
echo "pkg-config"
sudo  apt-get install -y pkg-config
echo "pip"
sudo  apt-get install -y python-pip
echo "pandas"
sudo apt-get install -y python-tk
echo "pandas"
pip install pandas
echo "emcee"
pip install emcee
echo "utils"
pip install utils
echo "torch"
pip install torch
pip install torchvision
echo "sklearn"
pip install sklearn
pip install scipy
# pip install scikit-image
echo "skimage"
sudo apt-get install -y python-skimage
echo "mnist"
pip install python-mnist
echo "matplotlib"
pip install 'matplotlib==2.2.2' --force-reinstall

cd $HOME/gopath/src/Biscotti/ML/Pytorch/data/mnist
python parse_mnist.py

cd $HOME/gopath/src/Biscotti/DistSys
mkdir LogFiles

go install
