#!/bin/bash -x
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters (expecting 1):"
    echo "[name of resource group]"
    exit
fi

rgroup=$1

vmprefix='bis'

let numVMs=20

for (( i = 0; i < numVMs; i++ )); do
	
	vmname=$vmprefix$i
	bash stop-vm.sh $rgroup $vmname

done
