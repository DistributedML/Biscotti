#!/bin/bash -x

filename=$1
hostfile="../azure-conf/$filename"
rgroup='biscotti'
vmprefix='bis'

let numVMs=20

for (( i = 0; i < numVMs; i++ )); do
	
	vmname=$vmprefix$i
	bash stop-vm.sh $rgroup $vmname

done
