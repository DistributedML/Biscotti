#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[r-group, numVMs, outFileName]"
    exit
fi

rgroup=$1   # resource group
let numVMs=$2   # number of VMs to fetch names of
outFileName=$3  # name of file to append to

#Hardcoded vmprefix and azure-conf path
vmprefix='bis' 
outFilePath='../azure-conf/'

#go to azure-conf dir
cd $outFilePath

# Purge file
> $outFileName

# fetch vm ip. Append to file
for (( i = 0; i < numVMs; i++ )); do

	vmname=$vmprefix$i # generate vm name
	
	vmip=$(az vm show --resource-group $rgroup --name $vmname --show-details \
		| grep -E 'publicIps' \
		| grep -E -o '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}')
	
	echo $vmip >> $outFileName	

done

