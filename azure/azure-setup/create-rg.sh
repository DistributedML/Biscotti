#!/bin/bash -x

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[rgName]"
    exit
fi

rgroup=$1
rglocation=$2

#create resource group
az group create --name $rgroup --location $rglocation
