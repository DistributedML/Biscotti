#!/bin/bash -x

hostfile='../azure-conf/hosts_diffDC'
rgroup='biscotti'
vmprefix='bis'

let numVMs=10

for (( i = 0; i < numVMs; i++ )); do
	
	vmname=$vmprefix$i
	bash stop-vm.sh $rgroup $vmname

done