#!/bin/bash

#install golang
## go to home directory
cd

# #download go binary
wget https://dl.google.com/go/go1.10.3.linux-amd64.tar.gz

# # #unzip and remove
sudo tar -C /usr/local -xzf go1.10.3.linux-amd64.tar.gz
rm go1.10.3.linux-amd64.tar.gz

# # #export path
export PATH=$PATH:/usr/local/go/bin
echo "" >> .profile
echo "#export go path" >> .profile
echo "export PATH=$PATH:/usr/local/go/bin" >> .profile

# # #make root directory and set GOPATH
sudo rm -rf gopath
mkdir -p gopath/src
export GOPATH=$HOME/gopath
echo "" >> .profile
echo "#set GOPATH" >> .profile
echo "export GOPATH=$HOME/gopath" >> .profile

# # #export workspace bin
export PATH=$PATH:$GOPATH/bin
echo "" >> .profile
echo "#set local bin" >> .profile
echo "export PATH=$PATH:$GOPATH/bin" >> .profile
# # #Go should be installed now

# # Pip doesn't work without this
echo "export LC_ALL=C" >> .profile

# Install Biscotti
echo "Installing Biscotti"
cd gopath/src
git clone https://github.com/DistributedML/Biscotti.git


source ~/.bashrc

# Install dependencies
# echo "Installing Dependencies"
go get github.com/DistributedClocks/GoVector/govec
go get github.com/kniren/gota/dataframe
go get github.com/sbinet/go-python
go get github.com/coniks-sys/coniks-go/crypto/vrf

cp -a $HOME/gopath/src/Biscotti/lib/gonum $HOME/gopath/src/github.com
cp -a $HOME/gopath/src/Biscotti/lib/dedis $HOME/gopath/src/github.com

cd github.com/dedis/kyber
go get -t ./... # install 3rd-party dependencies

# source ~/.profile

cd $HOME

# # #Installing python libraries
echo "Update"
sudo apt-get update
echo "pkg-config"
sudo  apt-get install -y pkg-config
echo "pip"
sudo  apt-get install -y python-pip
echo "python-tk"
sudo apt-get install -y python-tk
echo "skimage"
sudo apt-get install -y python-skimage
pip install -r requirements.txt

cd $HOME/gopath/src/Biscotti/ML/Pytorch/data/mnist
python parse_mnist.py

cd $HOME/gopath/src/Biscotti/DistSys
mkdir LogFiles

go install