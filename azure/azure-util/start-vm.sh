#!/bin/bash -x

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[r-group, vm-name]"
    exit
fi

rgroup=$1   # resource group
vm=$2       # VM name to start

echo 'Queueing up vm START with no-wait..'
az vm start --resource-group $rgroup --name $vm --no-wait