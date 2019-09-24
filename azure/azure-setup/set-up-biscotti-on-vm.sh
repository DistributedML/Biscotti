#!/bin/bash -x

#Assumption: You have ssh access to the machine using your public key.
#Assumption: Have the installation script in the same folder as this script.

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[username, ipaddr, installationScript]"
    exit
fi

uname=$1
ipaddr=$2
installscript=$3

scp $installscript $uname@$ipaddr:~
scp requirements.txt $uname@$ipaddr:~

echo $installscript

ssh -t $uname@$ipaddr "
	bash '${installscript}'
"