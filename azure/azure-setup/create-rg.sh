#!/bin/bash -x

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[rgName]"
    exit
fi

rgroup=$1
rglocation='westus'

#create resource group
az group create --name $rgroup --location $rglocation