#!/bin/bash -x

#Assumption: You have ssh access to all machines using your public key.
#Assumption: Text file containing all ip's is available

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[username, ipfilename, installscript]"
    exit
fi

uname=$1
ipfilename=$2
installscript=$3

ipfilepath='../azure-conf/'

hostfile=$ipfilepath$ipfilename

for ip in $(cat $hostfile);do

	bash set-up-biscotti-on-vm.sh $uname $ip $installscript

done