#!/bin/bash -x

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters (expecting 1):"
    echo "[name of resource group]"
    exit
fi

rgroup=$1
let numVMs=20
vmprefix='bis'

for (( i = 0; i < numVMs; i++ )); do

	vmname=$vmprefix$i

	bash start-vm.sh $rgroup $vmname
done
