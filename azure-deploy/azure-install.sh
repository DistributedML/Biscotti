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
mkdir gopath
export GOPATH=$HOME/gopath
echo "" >> .profile
echo "#set GOPATH" >> .profile
echo "export GOPATH=$HOME/go" >> .profile

#export workspace bin
export PATH=$PATH:$GOPATH/bin
echo "" >> .profile
echo "#set local bin" >> .profile
echo "export PATH=$PATH:$GOPATH/bin" >> .profile
#Go should be installed now

#clone the TorMentor Repository
echo "Installing Biscotti"
go get github.com/m-shayanshafi/simpleBlockChain

# I don't really like this requirement.... TODO fix it.
# export GOPATH=$HOME/go/src/github.com/m-shayanshafi/simpleBlockChain/DistSys

#Install dependencies
echo "Installing Dependencies"
go get github.com/DistributedClocks/GoVector/govec
go get github.com/kniren/gota/dataframe
go get github.com/sbinet/go-python
go get -u gonum.org/v1/gonum/mat
go get github.com/coniks-sys/coniks-go/crypto/vrf

##TODO Probably install python
echo "pkg-config"
apt-get install -y pkg-config
echo "pip"
apt-get install -y python-pip
echo "pandas"
pip install pandas
echo "emcee"
pip install emcee
echo "utils"
pip install utils

cd $HOME/gopath/src/github.com/m-shayanshafi/simpleBlockChain/DistSys 
go install
