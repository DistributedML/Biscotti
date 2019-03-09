#!/bin/bash -x

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "usage:"
    echo "[numVms, rgroup]"
    exit
fi

numVMs=$1    # Number of vms to deploy
rgroup=$2    # resource group in which to find the image/create VM
vmprefix='bis'

for (( i = 0; i < numVMs; i++ )); do

	vmname=$vmprefix$i
	bash start-vm.sh $rgroup $vmname

done
